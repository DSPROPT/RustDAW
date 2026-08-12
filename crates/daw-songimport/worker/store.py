"""Project store and manifest writer for the RustDAW song-import worker.

RustDAW is only an HTTP client; this module owns the on-disk contract it reads.
A finished project lives at ``<data_dir>/projects/<id>/`` and holds:

* ``project.json`` — the manifest RustDAW's ``SongManifest`` parses.
* ``stems/<name>.wav`` — one file per separated instrument.
* ``midi/song.mid`` — the transcription (optional).

The manifest schema is deliberately the subset RustDAW consumes, written in the
exact camelCase shape ``manifest.rs`` expects. Keep the two in step.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from pathlib import Path

# The six Demucs stems RustDAW knows how to order, plus the drum sub-kit names.
STEM_NAMES = ["drums", "bass", "other", "vocals", "guitar", "piano"]
DRUMKIT_NAMES = ["kick", "snare", "toms", "cymbals"]

DATA_DIR_NAME = "chords-extraction"


def data_dir() -> Path:
    """Resolve the worker's data directory, matching RustDAW's ``supervisor``.

    ``CHORDS_STUDIO_DATA`` wins; otherwise an existing installation is preferred
    wherever it sits, falling back to the platform's conventional location. The
    candidate order here mirrors ``data_dir_candidates`` in ``supervisor.rs`` so
    both programs always agree on where projects live.
    """
    override = os.environ.get("CHORDS_STUDIO_DATA")
    if override:
        return Path(override)
    home = Path.home()
    candidates: list[Path] = []
    if platform.system() == "Darwin":
        candidates.append(home / "Library" / "Application Support" / DATA_DIR_NAME)
    candidates.append(home / ".local" / "share" / DATA_DIR_NAME)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def projects_dir() -> Path:
    return data_dir() / "projects"


_SLUG = re.compile(r"[^a-z0-9]+")


def make_project_id(title: str | None) -> str:
    """A filesystem-safe, single-component project id, timestamp-prefixed.

    RustDAW rejects any id that is not a plain path component, so the slug is
    stripped to ``[a-z0-9-]`` and can never contain a separator.
    """
    import time

    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = (title or "untitled").lower()
    slug = _SLUG.sub("-", base).strip("-") or "untitled"
    return f"{stamp}-{slug[:48]}"


def project_dir(project_id: str) -> Path:
    root = projects_dir()
    path = (root / project_id).resolve()
    # Never let an id escape the store, mirroring RustDAW's own guard.
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"invalid project id: {project_id!r}")
    return path


def write_manifest(
    directory: Path,
    *,
    title: str | None,
    artist: str | None,
    style: str | None,
    source_url: str | None,
    duration: float | None,
    stems: dict[str, str],
    drumkit: dict[str, str] | None,
    midi: dict[str, str] | None,
    beat_grid: dict | None,
    stages: dict[str, dict] | None,
) -> Path:
    """Write ``project.json`` in the shape RustDAW's ``SongManifest`` reads."""
    manifest = {
        "title": title,
        "artist": artist,
        "style": style,
        "sourceUrl": source_url,
        "duration": duration,
        "files": {
            "stems": stems,
            "drumkit": drumkit or {},
            "midi": midi or {},
        },
        "beatGrid": beat_grid,
        "stages": stages or {},
    }
    path = directory / "project.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def read_manifest(directory: Path) -> dict:
    return json.loads((directory / "project.json").read_text())


def list_projects() -> list[dict]:
    """Summaries for every finished project, newest first.

    Shaped as RustDAW's ``ProjectSummary`` (camelCase, ``hasStems``).
    """
    root = projects_dir()
    if not root.is_dir():
        return []
    summaries: list[dict] = []
    for child in sorted(root.iterdir(), reverse=True):
        manifest_path = child / "project.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        stems = manifest.get("files", {}).get("stems", {})
        summaries.append(
            {
                "id": child.name,
                "title": manifest.get("title"),
                "artist": manifest.get("artist"),
                "style": manifest.get("style"),
                "duration": manifest.get("duration"),
                "hasStems": bool(stems),
            }
        )
    return summaries


def _self_test() -> int:
    """Write a synthetic project so the RustDAW wiring can be exercised offline.

    Produces silent stems, a plausible beat grid and an empty MIDI file — enough
    for RustDAW to import a session without any model download. Prints the id.
    """
    import math
    import struct
    import wave

    directory = project_dir(make_project_id("self test"))
    (directory / "stems").mkdir(parents=True, exist_ok=True)
    (directory / "midi").mkdir(parents=True, exist_ok=True)

    def tone_wav(path: Path, frequency: float, seconds: float = 4.0, rate: int = 44100) -> None:
        # A quiet sine (~-20 dBFS), well above RustDAW's -60 dB silence floor so
        # the stem is kept and is actually audible on import.
        amplitude = 3276
        frames = int(seconds * rate)
        samples = []
        for index in range(frames):
            value = int(amplitude * math.sin(2 * math.pi * frequency * index / rate))
            samples.append(value)  # left
            samples.append(value)  # right
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))

    # A distinct pitch per stem so they are audibly different on playback.
    stem_tones = {"drums": 110.0, "bass": 82.0, "other": 220.0, "vocals": 330.0}
    stems = {}
    for name, frequency in stem_tones.items():
        rel = f"stems/{name}.wav"
        tone_wav(directory / rel, frequency)
        stems[name] = rel

    # A steady 120 BPM 4/4 grid: beats every 0.5 s.
    beat_times = [round(index * 0.5, 3) for index in range(64)]
    write_manifest(
        directory,
        title="Self Test",
        artist="RustDAW",
        style="Test",
        source_url=None,
        duration=4.0,
        stems=stems,
        drumkit=None,
        midi=None,
        beat_grid={
            "beatTimes": beat_times,
            "beatsPerBar": 4,
            "downbeatIndex": 0,
            "source": "self-test",
        },
        stages={"download": {"status": "done"}, "separate": {"status": "done"}},
    )
    print(directory.name)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        raise SystemExit(_self_test())
    print(f"data dir: {data_dir()}")
    print(f"projects: {projects_dir()}")
