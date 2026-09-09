"""The song-separation pipeline: download, separate, transcribe, analyse.

Each heavy dependency (torch, demucs, basic-pitch, librosa, yt-dlp) is imported
inside the function that needs it, so the HTTP server and the ``store`` self-test
keep working before the models are installed, and a missing package surfaces as a
clear per-stage error instead of an import failure at start-up.

Transcription has two backends. MuScriptor is preferred when it is installed: it
knows what instrument it is listening to, so it names its own tracks and brings
back the drum kit basic-pitch has to leave out. It lives in its own virtualenv
(its NumPy 2 floor does not agree with the Demucs stack here) and is driven as a
subprocess, so nothing it needs is installed alongside these packages.
basic-pitch stays the fallback, and remains the only backend whose weights are
free of a non-commercial clause.

Both backends run once per separated stem rather than once over the mix. The
model decodes in five-second chunks and re-decides the instrument in every one,
so over a mix a single guitar comes back as three tracks that each stop where
the next begins — tracks that are mostly empty, and a part nobody played. A stem
settles that question before the model is asked, and MuScriptor is additionally
told which instruments the stem may contain.

The pipeline writes exactly the project layout ``store`` documents and RustDAW
reads. It does the separation and transcription; RustDAW converts the stems to
the session rate and builds the session.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from store import make_project_id, project_dir, write_manifest


def ffmpeg_binary() -> str:
    """Resolve the ffmpeg executable.

    A process started by a Finder-launched app inherits a minimal PATH without
    Homebrew, so a bare ``ffmpeg`` is not found even when installed. ``FFMPEG``
    overrides; otherwise the usual install locations are tried before the name.
    """
    override = os.environ.get("FFMPEG")
    if override:
        return override
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if Path(candidate).is_file():
            return candidate
    return "ffmpeg"


def to_wav(source: Path, into: Path) -> Path:
    """Decode any downloaded audio to a plain 44.1 kHz stereo WAV.

    Doing this once up front means neither Demucs nor librosa has to decode the
    source's codec (YouTube audio is usually Opus in a WebM), which is where the
    ``unsupported codec`` failures came from.
    """
    destination = into / "source.wav"
    result = subprocess.run(
        [
            ffmpeg_binary(), "-nostdin", "-y", "-i", str(source),
            "-ar", "44100", "-ac", "2", str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not destination.is_file():
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["ffmpeg failed"]
        raise RuntimeError(f"could not decode audio to WAV: {tail[0]}")
    return destination

# MuScriptor delays every note so that bar 1 starts on a real downbeat, and
# writes the amount into the MIDI as a marker with this prefix.
MUSCRIPTOR_BAR_OFFSET = "muscriptor:bar_offset="


def muscriptor_binary() -> str | None:
    """The MuScriptor CLI to drive, or ``None`` when it is not installed.

    ``MUSCRIPTOR_BIN`` wins, then the virtualenv ``install.sh --with-muscriptor``
    creates beside the worker's own, then the PATH — so an install made with uv
    or pipx is picked up without any of ours.
    """
    override = os.environ.get("MUSCRIPTOR_BIN")
    if override:
        return override if Path(override).is_file() else None
    from store import data_dir

    candidate = data_dir() / "venv-muscriptor" / "bin" / "muscriptor"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("muscriptor")


def muscriptor_token() -> str | None:
    """A HuggingFace token for the gated weights, when one is set aside here.

    ``HF_TOKEN`` in the environment wins and needs no help — the subprocess
    inherits it. Otherwise ``muscriptor.token`` beside the worker's projects is
    read, which is the only place a token can live and still be found: RustDAW
    starts the worker from a desktop session, where nothing exported in a shell
    ever arrives, and writing one into the machine's own ``hf auth login`` would
    change which HuggingFace account everything else on it uses.

    Returns ``None`` when the environment already carries one or there is no
    file, both of which mean the subprocess needs no extra environment.
    """
    if os.environ.get("HF_TOKEN"):
        return None
    from store import data_dir

    path = data_dir() / "muscriptor.token"
    if not path.is_file():
        return None
    return path.read_text().strip() or None


# The published variants worth running on a GPU, best first. Each is gated
# separately on HuggingFace, so a token that opened ``medium`` does not
# necessarily open ``large``; the first one whose weights this machine can
# actually fetch is the one used, and the answer is remembered for the process.
MUSCRIPTOR_MODELS = ("large", "medium")

_RESOLVED_MODEL: str | None = None


def muscriptor_model(device: str | None = None) -> str:
    """Which published variant to run.

    The largest one this machine can reach, because transcription quality is
    what the whole stage is for. Each variant is a separate gated repository, so
    ``large`` is tried first and ``medium`` — MuScriptor's own default — is used
    when its licence has not been accepted. Neither wants to run on a CPU: they
    decode autoregressively over five-second chunks, which without a GPU takes
    long enough to dominate the whole import, so ``small`` is used there.

    ``MUSCRIPTOR_MODEL`` overrides all of it.
    """
    override = os.environ.get("MUSCRIPTOR_MODEL")
    if override:
        return override
    if (device or torch_device()) == "cpu":
        return "small"
    global _RESOLVED_MODEL  # noqa: PLW0603 — one probe per process, not per stem
    if _RESOLVED_MODEL is None:
        _RESOLVED_MODEL = resolve_muscriptor_model()
    return _RESOLVED_MODEL


def resolve_muscriptor_model() -> str:
    """The best variant whose weights this machine can fetch.

    Asked by starting a transcription of a moment of silence: the CLI resolves
    and downloads the weights before it decodes anything, so a variant that is
    gated fails in about a second and one that is available costs only the
    download it was going to do anyway. There is no lighter question to ask —
    the weights being in the HuggingFace cache is not the same as the licence
    still being accepted.
    """
    binary = muscriptor_binary()
    if binary is None:
        return MUSCRIPTOR_MODELS[-1]
    token = muscriptor_token()
    with tempfile.TemporaryDirectory() as scratch:
        probe = Path(scratch) / "probe.wav"
        silence = subprocess.run(
            [
                ffmpeg_binary(), "-nostdin", "-y", "-f", "lavfi",
                "-i", "anullsrc=r=44100:cl=mono", "-t", "1", str(probe),
            ],
            capture_output=True,
            text=True,
        )
        if silence.returncode != 0:
            return MUSCRIPTOR_MODELS[-1]
        for model in MUSCRIPTOR_MODELS[:-1]:
            result = subprocess.run(
                [
                    binary, "transcribe", str(probe),
                    "--format", "midi",
                    "--output", str(Path(scratch) / "probe.mid"),
                    "--model", model,
                    "--device", "auto",
                    "--detect-tempo", "false",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "HF_TOKEN": token} if token else None,
            )
            if result.returncode == 0:
                return model
    return MUSCRIPTOR_MODELS[-1]


def remove_bar_offset(path: Path) -> None:
    """Move a MuScriptor transcription back onto audio time, in place.

    A MIDI file has no pickup measure — bar 1 starts at tick 0 — so the only way
    to put a bar line on the first downbeat is to delay the music, by up to a
    whole bar. RustDAW lays the transcription against the stems on a beat grid it
    detects itself, where that delay is not alignment but a bar of drift, so it
    comes back out. The marker records exactly how much, including the lag
    correction already applied to the model's onsets, so subtracting it leaves
    the notes where they sound in the audio.
    """
    import mido

    midi = mido.MidiFile(str(path))
    offset_seconds: float | None = None
    tempo = 500_000
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                tempo = message.tempo
            elif message.type == "marker" and message.text.startswith(MUSCRIPTOR_BAR_OFFSET):
                try:
                    offset_seconds = float(message.text.removeprefix(MUSCRIPTOR_BAR_OFFSET))
                except ValueError:
                    offset_seconds = None
    if not offset_seconds:
        return
    shift = round(offset_seconds * 1_000_000 * midi.ticks_per_beat / tempo)
    if shift <= 0:
        return

    for track in midi.tracks:
        absolute = 0
        moved: list[tuple[int, object]] = []
        for message in track:
            absolute += message.time
            if message.type == "marker" and message.text.startswith(MUSCRIPTOR_BAR_OFFSET):
                # Dropped with the shift it describes, so that reading this file
                # again does not delay it a second time.
                continue
            # Only the notes were delayed; the tempo, the time signature and the
            # end of track stay where they are. Clamping at zero costs at most
            # the lag correction (~25 ms) on the very first notes, which is the
            # only way any of them can land before the start.
            when = absolute if message.is_meta else max(0, absolute - shift)
            moved.append((when, message))
        # Stable, so a meta event and a note that now share a tick keep the
        # order they were written in.
        moved.sort(key=lambda pair: pair[0])
        previous = 0
        for when, message in moved:
            message.time = when - previous
            previous = when
        track[:] = [message for _, message in moved]
    midi.save(str(path))


# GM program numbers for the melodic stems basic-pitch transcribes. Drums are a
# kit and vocals are too noisy to transcribe usefully, so both are left to their
# stems. MuScriptor needs none of this: it names its own tracks and picks its own
# programs, drums included.
STEM_PROGRAMS = {
    "bass": 33,  # Electric Bass (finger)
    "piano": 0,  # Acoustic Grand Piano
    "guitar": 25,  # Acoustic Guitar (steel)
    "other": 48,  # String Ensemble 1
}

ProgressFn = Callable[[str, float, str], None]

# YouTube only serves media to a player client whose session has cleared its bot
# check, and the clients yt-dlp picks by default no longer clear it: extraction
# succeeds and the media URL then answers 403. Each entry is a ``player_client``
# extractor argument, tried in order until one yields a file. ``web_embedded`` is
# the client that currently works without an account; the empty entry falls back
# to yt-dlp's own defaults for when that stops being true, and the last one is a
# wider net. On non-YouTube sites the argument is ignored, so the first attempt
# is the only one that runs.
#
# Clearing the check means solving the client's JS challenges, which needs a
# JavaScript runtime and the EJS solver scripts — the ``deno`` and
# ``yt-dlp[default]`` requirements. Without them every client 403s.
PLAYER_CLIENTS = ("web_embedded", "", "tv,web_safari,mweb")


def _noop(stage: str, percent: float, message: str) -> None:  # pragma: no cover
    pass


def torch_device() -> str:
    """Best available compute device: CUDA on Linux GPUs, MPS on Apple Silicon,
    else CPU. Demucs and basic-pitch both run on any of them."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download(url: str, into: Path, on_progress: ProgressFn) -> tuple[Path, dict]:
    """Download the best audio track for ``url`` with yt-dlp.

    Returns the downloaded file and a metadata dict (title, artist, duration).
    """
    import yt_dlp

    on_progress("download", 0.0, "starting download")
    output_template = str(into / "source.%(ext)s")

    def hook(status: dict) -> None:
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            percent = 100.0 * done / total if total else 0.0
            on_progress("download", percent, "downloading audio")

    options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    failures = []
    for attempt, clients in enumerate(PLAYER_CLIENTS):
        # A failed attempt can leave a partial source.*.part behind, which would
        # otherwise be picked up as the download of the next attempt.
        for leftover in into.glob("source.*"):
            leftover.unlink()
        if attempt:
            on_progress("download", 0.0, "retrying with another player client")
        attempt_options = dict(options)
        if clients:
            attempt_options["extractor_args"] = {"youtube": {"player_client": clients.split(",")}}
        try:
            with yt_dlp.YoutubeDL(attempt_options) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as error:  # noqa: BLE001 — every attempt's reason is reported together
            failures.append(f"[{clients or 'default'}] {error}")
            continue
        files = list(into.glob("source.*"))
        if files:
            break
        failures.append(f"[{clients or 'default'}] no audio file was produced")
    else:
        raise RuntimeError("yt-dlp failed:\n" + "\n".join(failures))
    metadata = {
        "title": info.get("track") or info.get("title"),
        "artist": info.get("artist") or info.get("uploader"),
        "duration": info.get("duration"),
    }
    on_progress("download", 100.0, "download complete")
    return files[0], metadata


# Demucs always writes all six stems, so a song with no piano still gets a
# piano stem: a few seconds of leaked transients and nothing else. Peak level
# cannot tell that apart from real music — the leak clips just as high — so the
# test is mean level relative to the loudest stem in the same mix, which scales
# with the master's own loudness. 35 dB down is inaudible against the rest of
# the song and is well clear of a genuinely sparse part: a vocal that only
# enters for two phrases still lands around 25 dB down.
BLEED_FLOOR_DB = 35.0


def mean_volume_db(path: Path) -> float | None:
    """Mean (RMS) level of a WAV in dBFS, or ``None`` when it cannot be read.

    ``volumedetect`` reports it for the whole file in one pass, which is the
    cheapest measurement available here and needs nothing that is not already
    installed for the decode.
    """
    result = subprocess.run(
        [
            ffmpeg_binary(), "-nostdin", "-v", "info", "-i", str(path),
            "-af", "volumedetect", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    for line in reversed((result.stderr or "").splitlines()):
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].strip().removesuffix("dB").strip())
            except ValueError:
                return None
    return None


def drop_bleed_stems(levels: dict[str, float | None]) -> set[str]:
    """The stems that are leakage rather than an instrument that is playing.

    A stem whose level could not be measured is always kept: a parsing change
    must never silently lose part of a song.
    """
    measured = {name: level for name, level in levels.items() if level is not None}
    if not measured:
        return set()
    loudest = max(measured.values())
    return {
        name for name, level in measured.items() if loudest - level >= BLEED_FLOOR_DB
    }


def demucs_quality_args(device: str) -> list[str]:
    """Separation-quality arguments suited to the machine this runs on.

    ``--shifts`` averages the model over randomly shifted copies of the audio,
    which is the one knob that reliably raises separation quality; it multiplies
    the work by the shift count, so it is only worth having on a GPU. A wider
    ``--overlap`` costs far less and softens the seams between the windows
    Demucs stitches its output from, which is where the smearing that confuses
    transcription comes from.

    ``DEMUCS_SHIFTS`` and ``DEMUCS_OVERLAP`` override both.
    """
    shifts = os.environ.get("DEMUCS_SHIFTS") or ("2" if device != "cpu" else "0")
    overlap = os.environ.get("DEMUCS_OVERLAP") or "0.5"
    args = ["--overlap", overlap]
    if shifts != "0":
        args += ["--shifts", shifts]
    if device == "cpu":
        # Demucs is single-threaded per job; on this many cores the wall clock
        # is otherwise several times longer than it needs to be.
        args += ["-j", str(max(1, (os.cpu_count() or 2) // 2))]
    return args


def separate(source: Path, into: Path, on_progress: ProgressFn) -> dict[str, str]:
    """Run Demucs (``htdemucs_6s``) and copy the stems into ``into/stems``.

    Returns a ``{name: relative_path}`` map for the manifest. Silent or missing
    stems are dropped so an instrument that is not in the song does not appear.
    """
    import torch  # noqa: F401  (ensures a clear error if torch is absent)
    from demucs.separate import main as demucs_main

    device = torch_device()
    out_root = into / "_demucs"
    quality = demucs_quality_args(device)
    on_progress("separate", 0.0, f"separating stems on {device}")
    demucs_main(
        [
            "-n",
            "htdemucs_6s",
            "-d",
            device,
            "-o",
            str(out_root),
            "--filename",
            "{stem}.{ext}",
            *quality,
            str(source),
        ]
    )
    stems_dir = into / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}
    # `--filename {stem}.{ext}` drops the stems straight into the model folder
    # rather than a per-track subfolder, so collect them wherever they landed.
    for wav in sorted((out_root / "htdemucs_6s").rglob("*.wav")):
        destination = stems_dir / f"{wav.stem}.wav"
        shutil.copyfile(wav, destination)
        produced[wav.stem] = destination
    shutil.rmtree(out_root, ignore_errors=True)
    if not produced:
        raise RuntimeError("Demucs produced no stems")

    on_progress("separate", 90.0, "checking which instruments are playing")
    # The test is relative to the loudest stem, which is never 35 dB below
    # itself, so this can never drop every stem of a song.
    bleed = drop_bleed_stems({name: mean_volume_db(path) for name, path in produced.items()})
    stems: dict[str, str] = {}
    for name, path in produced.items():
        if name in bleed:
            path.unlink(missing_ok=True)
            continue
        stems[name] = f"stems/{name}.wav"
    if bleed:
        on_progress("separate", 95.0, f"not in this song: {', '.join(sorted(bleed))}")
    on_progress("separate", 100.0, "separation complete")
    return stems


def transcribe(
    source: Path, into: Path, stems: dict[str, str], on_progress: ProgressFn
) -> tuple[dict[str, str] | None, str | None]:
    """Transcribe the song into ``midi/song.mid``, best backend first.

    Returns the manifest ``midi`` map and the backend that produced it, so the
    import can say which one ran. A MuScriptor that is installed but fails —
    unaccepted model licence, no HuggingFace token, no room on the GPU — falls
    back to basic-pitch rather than losing the transcription altogether.
    """
    if muscriptor_binary() is not None:
        model = muscriptor_model()
        try:
            midi = transcribe_muscriptor(source, into, stems, on_progress)
        except Exception as error:  # noqa: BLE001 — the fallback is the point
            on_progress("transcribe", 0.0, f"muscriptor unavailable ({error}); using basic-pitch")
        else:
            if midi is not None:
                return midi, f"muscriptor {model}"
    return transcribe_basic_pitch(into, stems, on_progress), "basic-pitch"


# Which MuScriptor instrument groups can plausibly be in each Demucs stem.
# Passed as ``--instruments``, which forbids everything else from being decoded
# at all. A stem holds one instrument by construction, so telling the model that
# is the difference between a guitar track and a guitar that turns into a synth
# lead halfway through the song. ``other`` is deliberately absent and
# is decoded with no constraint at all: it is whatever Demucs could not name, so
# there is no answer to forbid, and forbidding one only moves the notes onto an
# instrument the model is still allowed to name. See
# :func:`drop_covered_instruments`.
STEM_INSTRUMENTS = {
    "drums": ["drums"],
    "bass": ["acoustic_bass", "electric_bass"],
    "guitar": ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
    "piano": ["acoustic_piano", "electric_piano", "organ"],
    "vocals": ["voice"],
}

# Stems to transcribe, in the order their tracks should appear.
STEM_TRANSCRIBE_ORDER = ("drums", "bass", "guitar", "piano", "other", "vocals")


def drop_covered_instruments(instruments: list, stems: dict[str, str]) -> list:
    """Remove from the ``other`` stem what another stem already transcribes.

    Demucs puts into ``other`` whatever it could not name, so the strings, horns
    and keys of a song are there — but so is the leakage from every stem beside
    it, and transcribing that is a second, worse copy of a part that already has
    its own track.

    The duplicates have to go afterwards rather than being forbidden up front.
    Telling the decoder it may not say "drums" does not stop it hearing the drum
    leakage; it makes it name that leakage something it is allowed to say, and a
    stem with a guitar and some spill came back as nine tracks of woodwind and
    brass with no guitar on any of them. Deciding after the fact costs nothing
    and cannot distort what the model heard.

    A stem dropped as bleed is not in ``stems``, so its instruments are not
    covered and stay here — which matters, because Demucs failing to isolate a
    guitar is exactly the case where the guitar is in ``other`` instead.
    """
    covered = {
        group
        for name, groups in STEM_INSTRUMENTS.items()
        if name in stems
        for group in groups
    }
    # The name alone decides, drum kit included: a kit here is dropped when the
    # drum stem is the one carrying it, and kept when that stem was bleed and
    # this is where the drums actually ended up.
    return [
        instrument
        for instrument in instruments
        # MuScriptor names a track after its instrument group, spaced rather
        # than underscored ("electric bass" for ``electric_bass``).
        if instrument.name.replace(" ", "_") not in covered
    ]

# An instrument holding less than this share of a stem's notes is a chunk the
# model labelled differently from its neighbours, not a part someone played.
# Only ``other`` is filtered this way; every other stem is collapsed to one
# track regardless.
FRAGMENT_SHARE = 0.02


def per_stem_transcription() -> bool:
    """Whether to transcribe each stem separately rather than the mix once.

    Per stem is the default because it is what removes the holes: the model
    decodes in five-second chunks and re-decides the instrument in each one, so
    over a mix a single guitar arrives as three tracks that each stop where the
    next begins. A stem answers that question before the model is asked. It
    costs one pass per stem instead of one in total, so ``MUSCRIPTOR_PER_STEM=0``
    goes back to the single pass.
    """
    return os.environ.get("MUSCRIPTOR_PER_STEM", "1") not in ("0", "false", "no")


# How strongly a decode is held to the audio it is listening to, for a stem that
# can only be one instrument. At the model's own default of 1.0 it drops
# passages it is unsure of: on a test song the vocal stem came back with 21
# seconds of clearly-sung audio transcribed as nothing. 1.5 brought that to 2
# seconds and invented not one note in a silent passage. Higher does not help
# and starts to cost: at 3.0 the same stem gained one more second of recall and
# hallucinated notes across 26 seconds where the stem is silent.
#
# It is deliberately used on the singing and nowhere else, because that is the
# only place it was measured to help. Guidance sharpens every decision the
# decoder makes, including which instrument it is hearing, so on a stem with a
# choice to make it sharpens that choice differently in each five-second chunk —
# which is the fragmentation this whole path exists to remove. Measured on the
# same song: the ``other`` stem went from one guitar track of 4166 notes to four
# tracks whose guitar had 898 notes and a two-minute hole, and the piano stem
# lost a quarter of its notes for no gain in what it found. Even the drum stem,
# which can only be one instrument and so cannot fragment, came back no better
# (1 second of missed playing became 3).
#
# ``MUSCRIPTOR_CFG`` overrides, and applies to every stem.
STEM_CFG_COEF = {"vocals": "1.5"}
DEFAULT_CFG_COEF = "1.0"


def cfg_coef(stem: str | None) -> str:
    """The guidance strength to decode ``stem`` with."""
    override = os.environ.get("MUSCRIPTOR_CFG")
    if override:
        return override
    return STEM_CFG_COEF.get(stem or "", DEFAULT_CFG_COEF)


def run_muscriptor(
    source: Path,
    destination: Path,
    instruments: list[str] | None,
    stem: str | None = None,
) -> None:
    """One MuScriptor pass, written to ``destination`` and put back on audio time.

    Raises with the model's own reason when the pass fails, so the caller can
    decide between skipping one stem and abandoning the backend.
    """
    binary = muscriptor_binary()
    if binary is None:
        raise RuntimeError("muscriptor is not installed")
    token = muscriptor_token()
    command = [
        binary, "transcribe", str(source),
        "--format", "midi",
        "--output", str(destination),
        "--model", muscriptor_model(),
        "--device", "auto",
        # The notes are played, not engraved, so they are never quantised;
        # the detected tempo still goes into the file, and best-effort keeps
        # a song with no steady tempo from failing the whole stage.
        "--detect-tempo", "best-effort",
        "--cfg-coef", cfg_coef(stem),
    ]
    if instruments:
        command += ["--instruments", ",".join(instruments)]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "HF_TOKEN": token} if token else None,
    )
    if result.returncode != 0 or not destination.is_file():
        destination.unlink(missing_ok=True)
        raise RuntimeError(_muscriptor_error(result.stderr))
    remove_bar_offset(destination)


def collapse_to_one_instrument(instruments: list, name: str) -> list:
    """Every note of a stem on one track, named after what it mostly is.

    A stem is one instrument, so the several tracks a chunk-by-chunk decode can
    produce for it are the same part split at the points where the model changed
    its mind. Joining them back is what leaves a guitar track with a guitar on it
    from the first bar to the last instead of three tracks with holes.
    """
    import pretty_midi

    populated = [instrument for instrument in instruments if instrument.notes]
    if not populated:
        return []
    dominant = max(populated, key=lambda instrument: len(instrument.notes))
    merged = pretty_midi.Instrument(
        program=dominant.program, is_drum=dominant.is_drum, name=dominant.name or name
    )
    for instrument in populated:
        merged.notes.extend(instrument.notes)
    merged.notes.sort(key=lambda note: note.start)
    return [merged]


def merge_named_instruments(instruments: list) -> list:
    """One track per named instrument, with the passing mislabels dropped.

    For the ``other`` stem, where several real instruments do share the file: the
    same name appearing twice is always the same part interrupted, and a name
    carrying a handful of notes across a whole song is a chunk that decoded
    differently from the ones around it.
    """
    import pretty_midi

    by_name: dict[tuple[str, bool], list] = {}
    for instrument in instruments:
        if instrument.notes:
            by_name.setdefault((instrument.name, instrument.is_drum), []).append(instrument)
    total = sum(len(group_member.notes) for group in by_name.values() for group_member in group)
    merged = []
    for (name, is_drum), group in by_name.items():
        notes = [note for instrument in group for note in instrument.notes]
        if total and len(notes) < FRAGMENT_SHARE * total:
            continue
        instrument = pretty_midi.Instrument(
            program=group[0].program, is_drum=is_drum, name=name
        )
        instrument.notes = sorted(notes, key=lambda note: note.start)
        merged.append(instrument)
    return merged


def transcribe_muscriptor(
    source: Path, into: Path, stems: dict[str, str], on_progress: ProgressFn
) -> dict[str, str] | None:
    """Transcribe the song with MuScriptor into ``midi/song.mid``.

    One pass per separated stem, each told which instruments it may contain, and
    the results merged into the single multi-track file RustDAW imports. Falls
    back to one pass over the mix when there are no stems to work from — which is
    also what the model was built for, but on a mix it re-decides the instrument
    every five seconds and the tracks come back full of holes.

    Returns ``None`` when MuScriptor is not installed.
    """
    if muscriptor_binary() is None:
        return None
    midi_dir = into / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    destination = midi_dir / "song.mid"
    model = muscriptor_model()

    targets = [name for name in STEM_TRANSCRIBE_ORDER if name in stems]
    if not per_stem_transcription() or not targets:
        on_progress("transcribe", 0.0, f"transcribing the mix with muscriptor ({model})")
        run_muscriptor(source, destination, None)
        on_progress("transcribe", 100.0, "transcription complete")
        return {"song": "midi/song.mid"}

    import pretty_midi

    parts: list = []
    tempo: float | None = None
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, name in enumerate(targets):
            on_progress(
                "transcribe",
                100.0 * index / len(targets),
                f"transcribing {name} with muscriptor ({model})",
            )
            part_path = Path(scratch) / f"{name}.mid"
            try:
                run_muscriptor(
                    into / stems[name], part_path, STEM_INSTRUMENTS.get(name), name
                )
                part = pretty_midi.PrettyMIDI(str(part_path))
            except Exception as error:  # noqa: BLE001 — one stem must not lose the rest
                failures.append(f"{name} ({error})")
                continue
            # Every stem is the same recording, so they all detect the same
            # tempo; the first one that found a real pulse settles the file.
            if tempo is None:
                found = part.get_tempo_changes()[1]
                if len(found) and abs(float(found[0]) - 120.0) > 0.01:
                    tempo = float(found[0])
            parts.extend(
                merge_named_instruments(drop_covered_instruments(part.instruments, stems))
                if name == "other"
                else collapse_to_one_instrument(part.instruments, name)
            )

    if not parts:
        # Every stem failing is one reason, not six — an unaccepted licence, no
        # token, no room on the GPU — so this goes up as a failed backend and
        # basic-pitch gets the song.
        reasons = "; ".join(failures)
        raise RuntimeError(
            f"no stem could be transcribed: {reasons}"
            if reasons
            else "no stem could be transcribed"
        )
    combined = pretty_midi.PrettyMIDI(initial_tempo=tempo or 120.0)
    combined.instruments.extend(parts)
    combined.write(str(destination))
    if failures:
        on_progress("transcribe", 100.0, f"not transcribed: {', '.join(failures)}")
    on_progress("transcribe", 100.0, "transcription complete")
    return {"song": "midi/song.mid"}


def _muscriptor_error(stderr: str | None) -> str:
    """The one line worth reporting out of a failed MuScriptor run.

    MuScriptor prints its own failures as ``Error: …`` and then, for the gated
    weights, several more lines of instructions — so the last line is a footnote
    ("see the README") and the first ``Error:`` is the actual reason. Anything
    that fails without one (a crash, a killed process) leaves a traceback, whose
    last line is the useful part instead.
    """
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    for line in lines:
        if line.startswith("Error: "):
            return line.removeprefix("Error: ")
    return lines[-1] if lines else "muscriptor failed"


def transcribe_basic_pitch(
    into: Path, stems: dict[str, str], on_progress: ProgressFn
) -> dict[str, str] | None:
    """Transcribe the melodic stems into one multi-track ``midi/song.mid``.

    Each stem becomes a named instrument with a General MIDI program, so RustDAW
    labels the tracks and the synth plays a sensible sound. Returns the manifest
    ``midi`` map, or ``None`` when nothing could be transcribed.
    """
    from basic_pitch.inference import predict
    import pretty_midi

    targets = [name for name in STEM_PROGRAMS if name in stems]
    if not targets:
        return None

    combined = pretty_midi.PrettyMIDI()
    for index, name in enumerate(targets):
        on_progress(
            "transcribe",
            100.0 * index / len(targets),
            f"transcribing {name}",
        )
        wav_path = into / stems[name]
        _model_output, midi_data, _note_events = predict(str(wav_path))
        for instrument in midi_data.instruments:
            instrument.program = STEM_PROGRAMS[name]
            instrument.is_drum = False
            instrument.name = name
            combined.instruments.append(instrument)

    if not combined.instruments:
        return None
    midi_dir = into / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    combined.write(str(midi_dir / "song.mid"))
    on_progress("transcribe", 100.0, "transcription complete")
    return {"song": "midi/song.mid"}


def analyse_beats(source: Path, on_progress: ProgressFn) -> dict | None:
    """Detect a beat grid with librosa, in the shape RustDAW's ``BeatGrid`` reads."""
    import librosa
    import numpy as np

    on_progress("analyze", 0.0, "detecting tempo")
    audio, sample_rate = librosa.load(str(source), mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sample_rate)
    # librosa >= 0.10 returns tempo as a length-1 array, not a scalar.
    tempo_bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    if len(beat_times) < 2:
        return None
    on_progress("analyze", 100.0, f"~{tempo_bpm:.0f} BPM")
    return {
        "beatTimes": [round(float(time), 4) for time in beat_times],
        "beatsPerBar": 4,
        "downbeatIndex": 0,
        "source": "librosa",
    }


def run(url: str, on_progress: ProgressFn | None = None) -> str:
    """Run the full pipeline for ``url`` and return the finished project id."""
    progress = on_progress or _noop
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        downloaded, metadata = download(url, scratch_dir, progress)
        return _process(downloaded, scratch_dir, metadata, url, progress)


def run_file(
    audio: Path, on_progress: ProgressFn | None = None, title: str | None = None
) -> str:
    """Run the pipeline for a file already on this machine.

    The same work as :func:`run` with the download stage already done, so a song
    that was never on the internet goes through exactly the path a link does.

    ``title`` is the name to file it under. It is passed in rather than taken
    from ``audio`` because an upload is stored under a name of the worker's own
    making, and the song the person chose is the one they want to see again.
    """
    progress = on_progress or _noop
    if not audio.is_file():
        raise RuntimeError(f"no such file: {audio}")
    progress("download", 100.0, "using the uploaded file")
    metadata = {"title": title or audio.stem, "artist": None, "duration": None}
    with tempfile.TemporaryDirectory() as scratch:
        return _process(audio, Path(scratch), metadata, None, progress)


def wav_duration(path: Path) -> float | None:
    """Length of a WAV in seconds, or ``None`` if it cannot be read.

    A link brings its duration along in the download metadata; a file has to be
    measured, and by this point it has already been decoded to a plain WAV, so
    the header is enough and nothing has to be loaded.
    """
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except Exception:  # noqa: BLE001 — a duration is a nicety, never a failure
        return None


def _process(
    downloaded: Path,
    scratch_dir: Path,
    metadata: dict,
    source_url: str | None,
    progress: ProgressFn,
) -> str:
    """Separate, transcribe and analyse one downloaded or uploaded song."""
    # Decode to WAV once so the codec never trips up Demucs or librosa.
    source = to_wav(downloaded, scratch_dir)

    directory = project_dir(make_project_id(metadata.get("title")))
    directory.mkdir(parents=True, exist_ok=True)
    if metadata.get("duration") is None:
        metadata["duration"] = wav_duration(source)

    stems = separate(source, directory, progress)
    try:
        midi, backend = transcribe(source, directory, stems, progress)
    except Exception as error:  # noqa: BLE001 — transcription is best-effort
        progress("transcribe", 100.0, f"transcription skipped: {error}")
        midi, backend = None, None
    try:
        beat_grid = analyse_beats(source, progress)
    except Exception as error:  # noqa: BLE001 — beat detection is best-effort
        progress("analyze", 100.0, f"tempo detection skipped: {error}")
        beat_grid = None

    progress("finalize", 50.0, "writing manifest")
    write_manifest(
        directory,
        title=metadata.get("title"),
        artist=metadata.get("artist"),
        style=None,
        source_url=source_url,
        duration=metadata.get("duration"),
        stems=stems,
        drumkit=None,
        midi=midi,
        beat_grid=beat_grid,
        stages={
            "download": {"status": "done"},
            "separate": {"status": "done"},
            "transcribe": {
                "status": "done" if midi else "skipped",
                **({"backend": backend} if midi and backend else {}),
            },
        },
    )
    progress("finalize", 100.0, "done")
    return directory.name
