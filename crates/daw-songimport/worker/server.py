"""Loopback HTTP server for RustDAW's song import.

Implements the exact endpoints RustDAW's ``client.rs`` calls:

* ``GET  /api/health``      -> ``{ok, cuda, models}``
* ``GET  /api/projects``    -> ``[ProjectSummary]``
* ``POST /api/jobs``        -> start a job for ``{url}``, returns a ``Job``
* ``GET  /api/jobs/<id>``   -> poll a ``Job``

Jobs run on background threads; their state lives in memory. The server binds to
127.0.0.1 only, so nothing is ever reachable off the machine. Only the standard
library is used here, so the server starts instantly even before the ML packages
are imported — those load lazily when the first job runs.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import store

DEFAULT_PORT = 8765

# Job registry. Guarded by _LOCK; job dicts are replaced wholesale, never mutated
# in place from two threads at once.
_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}

_DEVICE: str | None = None


def _device() -> str:
    global _DEVICE
    if _DEVICE is None:
        try:
            from pipeline import torch_device

            _DEVICE = torch_device()
        except Exception:  # noqa: BLE001 — health must never fail on a bad import
            _DEVICE = "cpu"
    return _DEVICE


def _demucs_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("demucs") is not None


def _new_job() -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "status": "queued",
        "stage": "",
        "percent": 0.0,
        "message": "",
        "error": None,
        "projectId": None,
    }


def _update(job_id: str, **fields) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(fields)


def _run_job(job_id: str, url: str) -> None:
    def on_progress(stage: str, percent: float, message: str) -> None:
        _update(job_id, status="running", stage=stage, percent=float(percent), message=message)

    try:
        import pipeline

        project_id = pipeline.run(url, on_progress)
        _update(job_id, status="done", percent=100.0, projectId=project_id, message="done")
    except Exception as error:  # noqa: BLE001 — any failure becomes a reported job error
        _update(job_id, status="error", error=str(error), message=str(error))


class Handler(BaseHTTPRequestHandler):
    # Quiet: RustDAW does not read this server's stdout.
    def log_message(self, *_args) -> None:  # noqa: D401
        pass

    def _send(self, code: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — required name
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            self._send(
                200,
                {
                    "ok": True,
                    "cuda": _device() == "cuda",
                    "models": {
                        "demucs": "htdemucs_6s" if _demucs_available() else None,
                        "drumsep": False,
                    },
                },
            )
        elif path == "/api/projects":
            self._send(200, store.list_projects())
        elif path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/") :]
            with _LOCK:
                job = _JOBS.get(job_id)
            if job is None:
                self._send(404, {"error": f"unknown job {job_id}"})
            else:
                self._send(200, job)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 — required name
        path = self.path.split("?", 1)[0]
        if path != "/api/jobs":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON body"})
            return
        url = (body.get("url") or "").strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            self._send(400, {"error": "only http(s) links are accepted"})
            return
        job = _new_job()
        with _LOCK:
            _JOBS[job["id"]] = job
        threading.Thread(target=_run_job, args=(job["id"], url), daemon=True).start()
        self._send(200, job)


def main() -> int:
    port = int(os.environ.get("CHORDS_STUDIO_PORT", DEFAULT_PORT))
    # Ensure the store exists so /api/projects works on a fresh install.
    store.projects_dir().mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"song-import worker listening on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
