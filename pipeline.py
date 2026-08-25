"""
Core pipeline for the Viral Clip Bay web app.

Same steps as the CLI agent (download -> transcribe -> Claude picks moments ->
ffmpeg cuts), refactored to report progress via a callback so the web server
can stream status to the browser.
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

ProgressCB = Callable[[str, float, str], None]  # (stage, 0-100 pct, message)


def _noop(stage: str, pct: float, message: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: List[Word] = field(default_factory=list)


@dataclass
class ClipPick:
    start: float
    end: float
    title: str
    hook: str
    reason: str


class PipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Step 1: Download
# ---------------------------------------------------------------------------


def download_video(url: str, workdir: Path, progress: ProgressCB = _noop) -> Path:
    progress("download", 0, "Downloading source video…")
    out_template = str(workdir / "source.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"Could not download this video: {result.stderr.strip()[-500:]}")

    video_files = [p for p in workdir.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm")]
    if not video_files:
        raise PipelineError("Download finished but no video file was produced.")
    progress("download", 100, "Video downloaded.")
    return video_files[0]


# ---------------------------------------------------------------------------
# Step 2: Transcribe
# ---------------------------------------------------------------------------


def extract_audio(video_path: Path, workdir: Path, progress: ProgressCB = _noop) -> Path:
    progress("extract_audio", 0, "Extracting audio…")
    audio_path = workdir / "audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"Could not extract audio: {result.stderr.strip()[-500:]}")
    progress("extract_audio", 100, "Audio extracted.")
    return audio_path


def transcribe(audio_path: Path, model_size: str, progress: ProgressCB = _noop) -> List[TranscriptSegment]:
    from faster_whisper import WhisperModel

    progress("transcribe", 0, f"Loading Whisper ({model_size})…")
    model = WhisperModel(model_size, compute_type="int8")
    segments_iter, info = model.transcribe(str(audio_path), word_timestamps=True)
    total_duration = max(info.duration, 1.0)

    result: List[TranscriptSegment] = []
    for seg in segments_iter:
        words = [Word(w.word.strip(), w.start, w.end) for w in (seg.words or [])]
        result.append(TranscriptSegment(seg.text.strip(), seg.start, seg.end, words))
        pct = min(99.0, (seg.end / total_duration) * 100)
        progress("transcribe", pct, f"Transcribing… {seg.end:.0f}s / {total_duration:.0f}s")

    progress("transcribe", 100, "Transcription complete.")
    return result


# ---------------------------------------------------------------------------
# Step 3: Ask Claude to pick clip-worthy moments
# ---------------------------------------------------------------------------


def _build_transcript_prompt(segments: List[TranscriptSegment]) -> str:
    return "\n".join(f"[{seg.start:07.2f} - {seg.end:07.2f}] {seg.text}" for seg in segments)


def pick_clips(segments: List[TranscriptSegment], num_clips: int,
                min_len: float, max_len: float, progress: ProgressCB = _noop) -> List[ClipPick]:
    import anthropic

    progress("select", 10, "Asking Claude to find the best moments…")
    client = anthropic.Anthropic()
    transcript_text = _build_transcript_prompt(segments)

    system_prompt = (
        "You are an expert short-form video producer who finds the most "
        "shareable, scroll-stopping moments in long-form video transcripts "
        "for TikTok/Reels/Shorts."
    )
    user_prompt = f"""Here is a timestamped transcript of a video (format: [start - end] text):

{transcript_text}

Pick the {num_clips} best standalone segments to cut into viral short-form clips.

Rules:
- Each clip must be self-contained (makes sense without the rest of the video).
- Each clip should be between {min_len:.0f} and {max_len:.0f} seconds long.
- Prefer moments with a strong hook in the first 3 seconds, an emotional peak,
  a surprising claim, a punchline, or a concrete actionable takeaway.
- Use exact timestamps from the transcript (do not invent times outside its range).
- Do not overlap clips.

Respond with ONLY a JSON array, no prose, no markdown fences, in this exact shape:
[
  {{"start": 12.3, "end": 58.1, "title": "short punchy title", "hook": "the first line/hook of the clip", "reason": "why this will perform well"}}
]
"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise PipelineError(f"Claude's response wasn't valid JSON: {e}") from e

    picks = [ClipPick(**c) for c in raw]
    progress("select", 100, f"Picked {len(picks)} clip(s).")
    return picks


# ---------------------------------------------------------------------------
# Step 4: Cut clips (+ optional vertical crop + captions)
# ---------------------------------------------------------------------------


def _srt_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def _build_srt(segments: List[TranscriptSegment], clip_start: float, clip_end: float) -> str:
    entries, idx, chunk = [], 1, []

    def flush():
        nonlocal idx
        if not chunk:
            return
        start, end = chunk[0].start - clip_start, chunk[-1].end - clip_start
        text = " ".join(w.text for w in chunk).strip()
        if text:
            entries.append(f"{idx}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
            idx += 1
        chunk.clear()

    for seg in segments:
        for w in seg.words:
            if w.end < clip_start or w.start > clip_end:
                continue
            chunk.append(w)
            if len(chunk) >= 4:
                flush()
    flush()
    return "\n".join(entries)


def cut_clip(video_path: Path, pick: ClipPick, index: int, workdir: Path,
             vertical: bool, captions: bool, segments: List[TranscriptSegment]) -> Path:
    raw_out = workdir / f"clip_{index}_raw.mp4"
    final_out = workdir / f"clip_{index}.mp4"
    duration = pick.end - pick.start

    cmd = ["ffmpeg", "-y", "-ss", str(pick.start), "-i", str(video_path), "-t", str(duration)]
    if vertical:
        cmd += ["-vf", "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=1080:1920"]
    cmd += ["-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(raw_out)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"ffmpeg failed cutting clip {index}: {result.stderr.strip()[-500:]}")

    if not captions:
        raw_out.rename(final_out)
        return final_out

    srt_path = workdir / f"clip_{index}.srt"
    srt_path.write_text(_build_srt(segments, pick.start, pick.end), encoding="utf-8")
    style = (
        "FontName=Arial Black,FontSize=14,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=80"
    )
    burn_cmd = [
        "ffmpeg", "-y", "-i", str(raw_out),
        "-vf", f"subtitles={srt_path.name}:force_style='{style}'",
        "-c:a", "copy", str(final_out),
    ]
    result = subprocess.run(burn_cmd, capture_output=True, text=True, cwd=workdir)
    if result.returncode != 0:
        raise PipelineError(f"ffmpeg failed burning captions on clip {index}: {result.stderr.strip()[-500:]}")
    raw_out.unlink(missing_ok=True)
    return final_out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(url: str, output_dir: Path, num_clips: int, min_len: float, max_len: float,
                  vertical: bool, captions: bool, whisper_model: str,
                  progress: ProgressCB = _noop) -> List[dict]:
    """Runs the full pipeline and returns a list of clip result dicts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = output_dir / "_work"
    workdir.mkdir(parents=True, exist_ok=True)

    video_path = download_video(url, workdir, progress)
    audio_path = extract_audio(video_path, workdir, progress)
    segments = transcribe(audio_path, whisper_model, progress)

    if not segments:
        raise PipelineError("No speech was detected in this video — nothing to clip.")

    picks = pick_clips(segments, num_clips, min_len, max_len, progress)

    results = []
    total = len(picks)
    for i, pick in enumerate(picks, start=1):
        progress("cut", ((i - 1) / total) * 100, f"Cutting clip {i} of {total}: {pick.title}")
        out_path = cut_clip(video_path, pick, i, workdir, vertical, captions, segments)
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", pick.title)[:40].strip("_") or f"clip_{i}"
        filename = f"clip_{i}_{safe_title}.mp4"
        final_dest = output_dir / filename
        out_path.rename(final_dest)
        results.append({
            "filename": filename,
            "title": pick.title,
            "hook": pick.hook,
            "reason": pick.reason,
            "start": pick.start,
            "end": pick.end,
            "duration": round(pick.end - pick.start, 1),
        })
        progress("cut", (i / total) * 100, f"Finished clip {i} of {total}.")

    import shutil
    shutil.rmtree(workdir, ignore_errors=True)
    progress("done", 100, "All clips ready.")
    return results
