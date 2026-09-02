"""The song-separation pipeline: download, separate, transcribe, analyse.

Each heavy dependency (torch, demucs, basic-pitch, librosa, yt-dlp) is imported
inside the function that needs it, so the HTTP server and the ``store`` self-test
keep working before the models are installed, and a missing package surfaces as a
clear per-stage error instead of an import failure at start-up.

Transcription has two backends. MuScriptor is preferred when it is installed: it
is multi-instrument, so one pass over the mix replaces one basic-pitch pass per
stem and brings back the drum kit the per-stem path has to leave out. It lives in
its own virtualenv (its NumPy 2 floor does not agree with the Demucs stack here)
and is driven as a subprocess, so nothing it needs is installed alongside these
packages. basic-pitch stays the fallback, and remains the only backend whose
weights are free of a non-commercial clause.

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


def muscriptor_model(device: str | None = None) -> str:
    """Which published variant to run.

    ``medium`` is MuScriptor's own default and wants a GPU. It decodes
    autoregressively over five-second chunks, which on a CPU-only machine takes
    long enough to dominate the whole import, so ``small`` is used there
    instead. ``MUSCRIPTOR_MODEL`` overrides both.
    """
    override = os.environ.get("MUSCRIPTOR_MODEL")
    if override:
        return override
    return "small" if (device or torch_device()) == "cpu" else "medium"


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


def separate(source: Path, into: Path, on_progress: ProgressFn) -> dict[str, str]:
    """Run Demucs (``htdemucs_6s``) and copy the stems into ``into/stems``.

    Returns a ``{name: relative_path}`` map for the manifest. Silent or missing
    stems are dropped so an instrument that is not in the song does not appear.
    """
    import torch  # noqa: F401  (ensures a clear error if torch is absent)
    from demucs.separate import main as demucs_main

    device = torch_device()
    out_root = into / "_demucs"
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
            str(source),
        ]
    )
    stems_dir = into / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    stems: dict[str, str] = {}
    # `--filename {stem}.{ext}` drops the stems straight into the model folder
    # rather than a per-track subfolder, so collect them wherever they landed.
    for wav in sorted((out_root / "htdemucs_6s").rglob("*.wav")):
        name = wav.stem
        destination = stems_dir / f"{name}.wav"
        shutil.copyfile(wav, destination)
        stems[name] = f"stems/{name}.wav"
    shutil.rmtree(out_root, ignore_errors=True)
    if not stems:
        raise RuntimeError("Demucs produced no stems")
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
            midi = transcribe_muscriptor(source, into, on_progress)
        except Exception as error:  # noqa: BLE001 — the fallback is the point
            on_progress("transcribe", 0.0, f"muscriptor unavailable ({error}); using basic-pitch")
        else:
            if midi is not None:
                return midi, f"muscriptor {model}"
    return transcribe_basic_pitch(into, stems, on_progress), "basic-pitch"


def transcribe_muscriptor(source: Path, into: Path, on_progress: ProgressFn) -> dict[str, str] | None:
    """Transcribe the whole mix with MuScriptor into ``midi/song.mid``.

    One pass over the mix, not one per stem: the model is multi-instrument, so it
    decides for itself which instruments are playing, names each track after one
    and writes the drum kit to channel 10 — all of which RustDAW's importer
    already reads. Returns ``None`` when MuScriptor is not installed.
    """
    binary = muscriptor_binary()
    if binary is None:
        return None
    model = muscriptor_model()
    midi_dir = into / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    destination = midi_dir / "song.mid"
    on_progress("transcribe", 0.0, f"transcribing with muscriptor ({model})")
    result = subprocess.run(
        [
            binary, "transcribe", str(source),
            "--format", "midi",
            "--output", str(destination),
            "--model", model,
            "--device", "auto",
            # The notes are played, not engraved, so they are never quantised;
            # the detected tempo still goes into the file, and best-effort keeps
            # a song with no steady tempo from failing the whole stage.
            "--detect-tempo", "best-effort",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not destination.is_file():
        destination.unlink(missing_ok=True)
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["muscriptor failed"]
        raise RuntimeError(tail[0])
    remove_bar_offset(destination)
    on_progress("transcribe", 100.0, "transcription complete")
    return {"song": "midi/song.mid"}


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
        # Decode to WAV once so the codec never trips up Demucs or librosa.
        source = to_wav(downloaded, scratch_dir)

        directory = project_dir(make_project_id(metadata.get("title")))
        directory.mkdir(parents=True, exist_ok=True)

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
            source_url=url,
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
