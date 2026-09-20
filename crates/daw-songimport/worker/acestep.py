"""ACE-Step behind the song-import worker: song generation and per-track renders.

ACE-Step 1.5 is the generative model: a whole song from a description and
lyrics, a stem re-rendered in another sound, an instrument pulled out of a mix,
or a new part written against it. It lives in its own checkout with its own
virtualenv — it pins torch 2.10 and brings vLLM, transformers and diffusers,
none of which the Demucs stack here wants — and it already ships an HTTP API
server. So this module never imports it. It starts ``acestep-api`` on loopback
the first time a job needs it, keeps it running, and talks to it the way
RustDAW talks to this worker: submit a task, poll it, fetch the file.

Two DiT models are involved. ``acestep-v15-turbo`` writes songs and covers in a
few seconds; the stem tasks (``extract``, ``lego``, ``complete``) only exist on
``acestep-v15-base``, so the server is started with on-demand model loading and
asked for the right one per job. A request for a model the server has not
loaded is otherwise answered by whatever *is* loaded, silently, which for an
extraction means a fresh song instead of a stem — so every result is checked
against the model that was asked for.

Only the standard library is used, like ``server.py``: this runs inside the
worker's own process, next to Demucs.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

from store import data_dir

DEFAULT_PORT = 8766
SONG_MODEL = "acestep-v15-turbo"
STEM_MODEL = "acestep-v15-base"

# Tasks the stem model alone knows. ``cover`` and ``text2music`` run on
# whichever model is loaded; these are refused unless the base model answers.
STEM_TASKS = {"extract", "lego", "complete"}

# The instruments ACE-Step can name in a stem task, in its own vocabulary
# (``acestep.constants.TRACK_NAMES``). RustDAW maps its stem names onto these.
TRACK_NAMES = [
    "vocals", "backing_vocals", "drums", "percussion", "bass", "guitar",
    "keyboard", "synth", "strings", "brass", "woodwinds", "fx",
]

ProgressFn = Callable[[str, float, str], None]

_LOCK = threading.Lock()
_SERVER: subprocess.Popen | None = None
# Where the log stood when the server was last started.
_LOG_START = 0


class AceStepError(RuntimeError):
    """A failure the user can act on: not installed, would not start, refused."""


# --- Installation -------------------------------------------------------------


def acestep_home() -> Path:
    """The ACE-Step checkout ``install.sh --with-ace-step`` made, or its override."""
    override = os.environ.get("ACESTEP_HOME")
    if override:
        return Path(override)
    return data_dir() / "ace-step"


def acestep_python() -> Path | None:
    """The interpreter of ACE-Step's own virtualenv, when it is installed."""
    for candidate in (acestep_home() / ".venv" / "bin" / "python", acestep_home() / "venv" / "bin" / "python"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def is_installed() -> bool:
    return acestep_python() is not None


def port() -> int:
    return int(os.environ.get("CHORDS_STUDIO_ACESTEP_PORT", DEFAULT_PORT))


def base_url() -> str:
    return f"http://127.0.0.1:{port()}"


def downloaded_models() -> list[str]:
    """DiT checkpoints already on disk, so the health report can say what a
    job will and will not have to download first."""
    checkpoints = acestep_home() / "checkpoints"
    if not checkpoints.is_dir():
        return []
    return sorted(
        child.name
        for child in checkpoints.iterdir()
        if child.is_dir() and child.name.startswith("acestep-v15-")
    )


def describe() -> dict | None:
    """What ``/api/health`` reports. Cheap on purpose: nothing is started."""
    if not is_installed():
        return None
    return {
        "home": str(acestep_home()),
        "running": _is_answering(),
        "models": downloaded_models(),
        "songModel": song_model(),
        "stemModel": STEM_MODEL,
        "tracks": TRACK_NAMES,
    }


def song_model() -> str:
    return os.environ.get("ACESTEP_MODEL") or SONG_MODEL


# --- The server ---------------------------------------------------------------


def _is_answering() -> bool:
    try:
        with urllib.request.urlopen(f"{base_url()}/health", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _log_path() -> Path:
    path = data_dir() / "logs" / "acestep.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _last_log_line(since: int = 0) -> str:
    """The server's latest line of output, ignoring anything before ``since``
    bytes — the log is kept across starts, and an earlier run's last words
    are not this one's progress."""
    try:
        with _log_path().open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size <= since:
                return ""
            handle.seek(max(since, size - 4096))
            lines = handle.read().decode("utf-8", "replace").strip().splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(("\r", "\x1b")):
            return line[:160]
    return ""


def _stop_server() -> None:
    global _SERVER
    server = _SERVER
    _SERVER = None
    if server is None or server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()


def ensure_server(on_progress: ProgressFn) -> None:
    """Make sure ``acestep-api`` is answering, starting it if it is not.

    The first start is slow twice over: the checkpoints (several GB) download
    into ``<home>/checkpoints`` and then load. So this waits a long time by
    default (``ACESTEP_START_TIMEOUT``, an hour), reporting the server's own
    last log line so the wait is not a blank spinner, and gives up at once if
    the process dies.
    """
    global _SERVER, _LOG_START
    with _LOCK:
        if _is_answering():
            return
        python = acestep_python()
        if python is None:
            raise AceStepError(
                "ACE-Step is not installed. Run "
                "`crates/daw-songimport/worker/install.sh --with-ace-step` once."
            )
        home = acestep_home()
        if _SERVER is None or _SERVER.poll() is not None:
            env = dict(os.environ)
            env.update(
                {
                    # Checkpoints resolve under the project root; keep them
                    # beside the checkout rather than wherever the worker
                    # happened to be started from.
                    "ACESTEP_PROJECT_ROOT": str(home),
                    "ACESTEP_API_HOST": "127.0.0.1",
                    "ACESTEP_API_PORT": str(port()),
                    "ACESTEP_CONFIG_PATH": song_model(),
                    # One worker each is what makes swapping models per job
                    # safe; the server refuses on-demand loads otherwise.
                    "ACESTEP_QUEUE_WORKERS": "1",
                    "ACESTEP_API_WORKERS": "1",
                    "ACESTEP_ON_DEMAND_MODEL_LOAD": "true",
                    # Between jobs the models move to system RAM, so Demucs and
                    # the transcriber get the GPU back when a generated song
                    # goes on to be separated.
                    "ACESTEP_OFFLOAD_TO_CPU": env.get("ACESTEP_OFFLOAD_TO_CPU", "true"),
                    "ACESTEP_TMPDIR": str(home / ".cache" / "acestep" / "tmp"),
                    # So the log shows where a slow start has got to.
                    "PYTHONUNBUFFERED": "1",
                }
            )
            log = _log_path().open("ab")
            _LOG_START = log.tell()
            on_progress("acestep", 0.0, "starting ACE-Step (first start downloads its models, ~6 GB)")
            try:
                _SERVER = subprocess.Popen(
                    [str(python), "-m", "acestep.api_server"],
                    cwd=str(home),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            except OSError as error:
                raise AceStepError(f"ACE-Step could not be started: {error}") from error
            atexit.register(_stop_server)

        timeout = float(os.environ.get("ACESTEP_START_TIMEOUT") or 3600)
        started = time.monotonic()
        while time.monotonic() - started < timeout:
            if _is_answering():
                on_progress("acestep", 5.0, "ACE-Step is ready")
                return
            if _SERVER.poll() is not None:
                raise AceStepError(
                    f"ACE-Step exited while starting (code {_SERVER.returncode}); "
                    f"see {_log_path()}. Last line: {_last_log_line(_LOG_START)}"
                )
            elapsed = int(time.monotonic() - started)
            detail = _last_log_line(_LOG_START)
            on_progress(
                "acestep",
                min(4.0, elapsed / 60.0),
                f"loading ACE-Step ({elapsed // 60}:{elapsed % 60:02d}) {detail}".rstrip(),
            )
            time.sleep(2.0)
        raise AceStepError(f"ACE-Step did not answer within {int(timeout)} s; see {_log_path()}")


# --- HTTP ---------------------------------------------------------------------


def _multipart(fields: dict[str, str], files: dict[str, Path]) -> tuple[bytes, str]:
    """Encode text fields and files as ``multipart/form-data``."""
    boundary = f"----RustDAWWorker{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode()
        body += str(value).encode()
        body += b"\r\n"
    for name, path in files.items():
        filename = path.name.replace('"', "_")
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        ).encode()
        body += path.read_bytes()
        body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _call(path: str, payload: dict | None = None, *, files: dict[str, Path] | None = None, timeout: float = 120) -> dict:
    """POST to ACE-Step and unwrap its ``{data, code, error}`` envelope."""
    url = f"{base_url()}{path}"
    if files:
        body, content_type = _multipart({k: v for k, v in (payload or {}).items() if v is not None}, files)
    else:
        body = json.dumps(payload or {}).encode()
        content_type = "application/json"
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            envelope = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:400]
        raise AceStepError(f"ACE-Step refused {path}: HTTP {error.code} {detail}") from error
    except (urllib.error.URLError, OSError) as error:
        raise AceStepError(f"ACE-Step is not answering on {path}: {error}") from error
    if envelope.get("code") not in (None, 200) or envelope.get("error"):
        raise AceStepError(f"ACE-Step rejected {path}: {envelope.get('error') or envelope.get('code')}")
    return envelope.get("data") or {}


def _release(fields: dict, files: dict[str, Path] | None) -> str:
    data = _call("/release_task", fields, files=files, timeout=600)
    task_id = data.get("task_id")
    if not task_id:
        raise AceStepError(f"ACE-Step accepted the task but returned no id: {data}")
    return task_id


def _wait(task_id: str, on_progress: ProgressFn, stage: str, label: str) -> dict:
    """Poll one task until it settles and return its first result entry."""
    started = time.monotonic()
    while True:
        rows = _call("/query_result", {"task_id_list": [task_id]})
        row = rows[0] if isinstance(rows, list) and rows else {}
        status = row.get("status")
        raw = row.get("result") or "[]"
        try:
            results = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            results = []
        first = results[0] if isinstance(results, list) and results else {}
        if status == 1:
            if not first:
                raise AceStepError("ACE-Step finished without a result")
            return first
        if status == 2:
            raise AceStepError(f"ACE-Step failed: {first.get('error') or row.get('progress_text') or 'unknown error'}")
        # A model being swapped in or a download shows up here; a running
        # task reports its own stage and fraction.
        elapsed = int(time.monotonic() - started)
        fraction = float(first.get("progress") or 0.0)
        detail = (row.get("progress_text") or first.get("stage") or "").strip()[:120]
        percent = 10.0 + 80.0 * fraction if fraction > 0 else min(90.0, 10.0 + elapsed / 3.0)
        on_progress(stage, percent, f"{label} ({elapsed // 60}:{elapsed % 60:02d}) {detail}".rstrip())
        time.sleep(2.0)


def _download(file_url: str, destination: Path) -> Path:
    """Fetch a result file. ``file`` in a result is ``/v1/audio?path=…``."""
    url = file_url if file_url.startswith("http") else f"{base_url()}{file_url}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=600) as response, destination.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
    except (urllib.error.URLError, OSError) as error:
        raise AceStepError(f"could not fetch the generated audio: {error}") from error
    if destination.stat().st_size == 0:
        raise AceStepError("the generated audio is empty")
    return destination


def _check_model(result: dict, wanted: str | None) -> None:
    """Refuse a result produced by the wrong model.

    A stem task answered by the song model is a whole new song, not a stem,
    and would be imported as one without this.
    """
    if not wanted:
        return
    used = (result.get("dit_model") or "").strip()
    if used and used != wanted:
        raise AceStepError(
            f"ACE-Step ran this on {used} rather than {wanted}; the stem tasks need "
            f"the base model. Run `acestep-download {wanted}` in the ACE-Step checkout, "
            "or check the ACE-Step log."
        )


# --- Tasks --------------------------------------------------------------------


def _steps_for(model: str) -> dict:
    """Sampling settings per model family: turbo is distilled to eight steps;
    base wants more, with guidance."""
    if "turbo" in model:
        return {"inference_steps": 8}
    return {"inference_steps": int(os.environ.get("ACESTEP_BASE_STEPS") or 50), "guidance_scale": 7.0}


def generate_song(
    *,
    prompt: str,
    lyrics: str,
    duration: float,
    bpm: int | None,
    key_scale: str | None,
    time_signature: str | None,
    language: str | None,
    destination: Path,
    on_progress: ProgressFn,
) -> tuple[Path, dict]:
    """Write a whole song from a description and return it with its metadata.

    ``lyrics`` empty means an instrumental. ``bpm``/``key_scale`` left out are
    chosen by the model, and come back in the metadata so the import can
    seed its own detection with them.
    """
    ensure_server(on_progress)
    model = song_model()
    fields: dict = {
        "prompt": prompt,
        "lyrics": lyrics or "[Instrumental]",
        "audio_duration": float(duration),
        "audio_format": "wav",
        "batch_size": 1,
        "model": model,
        "thinking": (os.environ.get("ACESTEP_THINKING") or "1") not in ("0", "false", "no"),
        **_steps_for(model),
    }
    if bpm:
        fields["bpm"] = int(bpm)
    if key_scale:
        fields["key_scale"] = key_scale
    if time_signature:
        fields["time_signature"] = str(time_signature)
    if language:
        fields["vocal_language"] = language
    on_progress("generate", 8.0, f"asking {model} for {int(duration)} s of music")
    result = _wait(_release(fields, None), on_progress, "generate", "writing the song")
    _check_model(result, model)
    on_progress("generate", 92.0, "fetching the song")
    path = _download(result.get("file") or "", destination)
    metas = result.get("metas") or {}
    metas.update({"model": result.get("dit_model") or model, "seed": result.get("seed_value")})
    on_progress("generate", 100.0, "song generated")
    return path, metas


def render_track(
    *,
    source: Path,
    task: str,
    track: str | None,
    prompt: str,
    strength: float,
    destination: Path,
    on_progress: ProgressFn,
) -> tuple[Path, dict]:
    """Run one of the source-audio tasks on ``source`` and write the result.

    ``task`` is ACE-Step's own name: ``cover`` re-renders the audio it is
    given in the sound the prompt asks for (``strength`` says how closely the
    part is kept); ``extract`` pulls the ``track`` out of a mix; ``lego``
    writes a new ``track`` that fits the mix. The last two need the base model
    and are refused if any other one answered.
    """
    if task not in ("cover", "extract", "lego", "complete"):
        raise AceStepError(f"unknown ACE-Step task {task!r}")
    if track is not None and track not in TRACK_NAMES:
        raise AceStepError(f"ACE-Step cannot name a {track!r} track; choose one of {', '.join(TRACK_NAMES)}")
    ensure_server(on_progress)
    model = STEM_MODEL if task in STEM_TASKS else song_model()
    fields: dict = {
        "task_type": task,
        "prompt": prompt,
        "audio_format": "wav",
        "batch_size": 1,
        "model": model,
        "thinking": False,
        "audio_cover_strength": max(0.0, min(1.0, float(strength))),
        **_steps_for(model),
    }
    if track:
        fields["track_name"] = track
    label = {
        "cover": "re-rendering the track",
        "extract": f"extracting the {track or 'track'}",
        "lego": f"writing a {track or 'new'} part",
        "complete": "completing the arrangement",
    }[task]
    on_progress("render", 8.0, f"sending {source.name} to {model}")
    result = _wait(_release(fields, {"src_audio": source}), on_progress, "render", label)
    _check_model(result, model)
    on_progress("render", 92.0, "fetching the render")
    path = _download(result.get("file") or "", destination)
    info = {"model": result.get("dit_model") or model, "seed": result.get("seed_value"), "task": task}
    on_progress("render", 100.0, "render complete")
    return path, info
