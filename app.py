"""
Viral Clip Bay — web app server.

Serves the frontend (static/index.html) and a small JSON API that runs the
clipping pipeline in a background thread per job, so the browser can poll
for progress and then play/download the finished clips.

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."
    uvicorn app:app --reload
Then open http://127.0.0.1:8000
"""

import os
import re
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import PipelineError, run_pipeline

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Viral Clip Bay")

# In-memory job store. Fine for a single-user local tool; swap for redis/db
# if you ever need multiple concurrent users or persistence across restarts.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

YOUTUBE_URL_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[\w-]+"
)


class CreateJobRequest(BaseModel):
    url: str
    num_clips: int = Field(default=3, ge=1, le=6)
    min_len: float = Field(default=20, ge=5, le=300)
    max_len: float = Field(default=75, ge=5, le=300)
    vertical: bool = True
    captions: bool = True
    whisper_model: str = Field(default="small")


def _update_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _run_job(job_id: str, req: CreateJobRequest) -> None:
    job_dir = OUTPUT_DIR / job_id

    def progress(stage: str, pct: float, message: str) -> None:
        _update_job(job_id, stage=stage, progress=round(pct, 1), message=message)

    try:
        _update_job(job_id, status="running")
        clips = run_pipeline(
            url=req.url,
            output_dir=job_dir,
            num_clips=req.num_clips,
            min_len=req.min_len,
            max_len=req.max_len,
            vertical=req.vertical,
            captions=req.captions,
            whisper_model=req.whisper_model,
            progress=progress,
        )
        _update_job(job_id, status="done", progress=100, clips=clips, message="All clips ready.")
    except PipelineError as e:
        _update_job(job_id, status="error", error=str(e))
    except Exception as e:  # noqa: BLE001 - surface anything unexpected to the UI too
        _update_job(job_id, status="error", error=f"Unexpected error: {e}")


@app.post("/api/jobs")
def create_job(req: CreateJobRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(500, "Server is missing ANTHROPIC_API_KEY. Set it and restart the server.")
    if not YOUTUBE_URL_RE.match(req.url.strip()):
        raise HTTPException(400, "That doesn't look like a YouTube video URL.")
    if req.min_len >= req.max_len:
        raise HTTPException(400, "Min length must be less than max length.")

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued", "stage": "queued", "progress": 0.0,
            "message": "Queued…", "clips": None, "error": None,
        }

    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return job


@app.get("/api/clips/{job_id}/{filename}")
def get_clip(job_id: str, filename: str):
    # filename comes only from our own generated results, but guard path traversal anyway.
    safe_name = Path(filename).name
    path = OUTPUT_DIR / job_id / safe_name
    if not path.is_file():
        raise HTTPException(404, "Clip not found.")
    return FileResponse(path, media_type="video/mp4", filename=safe_name)


# Serve the frontend last so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
