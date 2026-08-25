# Viral Clip Bay

A local web app version of the clip agent: paste a YouTube link in the
browser, watch it download → transcribe → get analyzed by Claude → cut into
clips, then preview and download each one — no command line needed after
setup.

Everything runs **on your machine**. The video and audio never leave your
computer; only the plain-text transcript is sent to Claude to pick the
moments worth cutting.

## Setup

You'll need [ffmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`:

```bash
# macOS
brew install ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
# Windows: download from ffmpeg.org and add to PATH
```

Then, from this folder (Python 3.9+):

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # get one at console.anthropic.com
```

## Run it

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

Paste a YouTube URL, adjust the settings if you want (number of clips, clip
length range, vertical crop, burned-in captions, transcription model), and
click **Roll clips**. You'll see a live status bar as it moves through
Download → Transcribe → Select → Cut, then a "contact sheet" of finished
clips you can preview and download.

## How it's built

- `app.py` — FastAPI server. `POST /api/jobs` kicks off a background thread
  per job; the browser polls `GET /api/jobs/{id}` for progress and reads
  finished clips from `GET /api/clips/{id}/{filename}`.
- `pipeline.py` — the actual pipeline (download with `yt-dlp`, transcribe
  with `faster-whisper`, pick moments with Claude, cut with `ffmpeg`),
  reporting progress through a callback.
- `static/index.html` — the single-page frontend (vanilla HTML/CSS/JS, no
  build step).
- `output/{job_id}/` — where finished clips land on disk; intermediate
  files (source video, audio, subtitles) are deleted once a job finishes.

It's an in-memory job store, so it's built for one person running it
locally, not for deploying as a multi-user public service — restarting the
server clears job history (finished clip files on disk are untouched).

## Notes & limitations

- **Rights**: only run this on videos you own or have permission to re-cut
  and repost — downloading and republishing someone else's video can
  violate YouTube's Terms of Service and copyright law.
- **First run is slower** — `faster-whisper` downloads its model weights the
  first time you use a given size.
- **Vertical crop** is a simple center-crop, not face-tracking.
- **One job at a time is easiest to reason about** — the app *can* run
  multiple jobs concurrently (each gets its own thread and folder), but on a
  laptop CPU, transcription is the bottleneck, so clips will queue up slowly
  if you fire off several at once.
- Want it reachable from your phone on the same Wi-Fi? Run
  `uvicorn app:app --host 0.0.0.0 --reload` and open
  `http://<your-computer's-LAN-IP>:8000` from your phone's browser.
