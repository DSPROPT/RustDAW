"""The song-separation pipeline: download, separate, transcribe, analyse.

Each heavy dependency (torch, demucs, basic-pitch, librosa, yt-dlp) is imported
inside the function that needs it, so the HTTP server and the ``store`` self-test
keep working before the models are installed, and a missing package surfaces as a
clear per-stage error instead of an import failure at start-up.

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

# GM program numbers for the melodic stems we transcribe. Drums are a kit and
# vocals are too noisy to transcribe usefully, so both are left to their stems.
STEM_PROGRAMS = {
    "bass": 33,  # Electric Bass (finger)
    "piano": 0,  # Acoustic Grand Piano
    "guitar": 25,  # Acoustic Guitar (steel)
    "other": 48,  # String Ensemble 1
}

ProgressFn = Callable[[str, float, str], None]


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
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    files = list(into.glob("source.*"))
    if not files:
        raise RuntimeError("yt-dlp produced no audio file")
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


def transcribe(into: Path, stems: dict[str, str], on_progress: ProgressFn) -> dict[str, str] | None:
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
            midi = transcribe(directory, stems, progress)
        except Exception as error:  # noqa: BLE001 — transcription is best-effort
            progress("transcribe", 100.0, f"transcription skipped: {error}")
            midi = None
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
                "transcribe": {"status": "done" if midi else "skipped"},
            },
        )
        progress("finalize", 100.0, "done")
        return directory.name
