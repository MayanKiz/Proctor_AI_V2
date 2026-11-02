from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from eye_tracker import track_eye
from face_spoofing import detect_spoofing
from head_pose_estimation import detect_head_pose
from mouth_opening_detector import mouth_opening_detector
from person_and_phone import detect_phone_and_person

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"

SESSION_STATE = {
    "status": "idle",
    "pid": None,
    "started_at": None,
    "logs": ["System is ready. Start a live monitoring session from the website."],
    "process": None,
}

app = FastAPI(title="Mayank Proctoring AI")

MAYANK_BANNER = r"""
██████╗  █████╗ ██╗   ██╗ █████╗ ███╗   ██╗██╗  ██╗
██╔════╝ ██╔══██╗╚██╗ ██╔╝██╔══██╗████╗  ██║██║ ██╔╝
██║  ███╗███████║ ╚████╔╝ ███████║██╔██╗ ██║█████╔╝
██║   ██║██╔══██║  ╚██╔╝  ██╔══██║██║╚██╗██║██╔═██╗
╚██████╔╝██║  ██║   ██║   ██║  ██║██║ ╚████║██║  ██╗
 ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
"""


class StartSessionRequest(BaseModel):
    camera_index: int = 0
    bot_token: str = ""
    chat_id: str = ""


class TelegramRequest(BaseModel):
    bot_token: str = ""
    chat_id: str = ""


def add_log(message: str) -> None:
    if not message:
        return
    SESSION_STATE["logs"].append(message)
    if len(SESSION_STATE["logs"]) > 150:
        SESSION_STATE["logs"] = SESSION_STATE["logs"][-150:]


def read_process_output(process: subprocess.Popen) -> None:
    if process.stdout is None:
        return
    try:
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                add_log(line)
    finally:
        process.stdout.close()


def _resolve_telegram_token(token_value: str) -> str:
    token = (token_value or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    if token.lower().startswith(("your_", "paste_", "<")):
        return ""
    return token


def _resolve_telegram_chat_id(chat_id_value: str) -> str:
    chat_id = (chat_id_value or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
    if chat_id.lower().startswith(("your_", "paste_", "<")):
        return ""
    return chat_id


@app.on_event("startup")
def show_startup_banner() -> None:
    print(MAYANK_BANNER, flush=True)


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "session": SESSION_STATE["status"]}


@app.get("/api/status")
def get_status() -> dict:
    return {
        "status": SESSION_STATE["status"],
        "pid": SESSION_STATE["pid"],
        "started_at": SESSION_STATE["started_at"],
        "logs": SESSION_STATE["logs"],
    }


@app.post("/api/test-telegram")
def test_telegram(payload: TelegramRequest) -> dict:
    bot_token = _resolve_telegram_token(payload.bot_token)
    chat_id = _resolve_telegram_chat_id(payload.chat_id)

    if not bot_token or not chat_id:
        raise HTTPException(status_code=400, detail="Telegram bot token and chat ID are required.")

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": chat_id, "text": "✅ Proctor AI test message from the website backend."},
            timeout=15,
        )
        response.raise_for_status()
        payload_json = response.json()
        return {
            "ok": bool(payload_json.get("ok")),
            "message": payload_json.get("description") or "telegram message sent",
            "message_id": (payload_json.get("result") or {}).get("message_id"),
        }
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Telegram request failed: {exc}") from exc


@app.post("/api/start-check")
def start_live_check(payload: StartSessionRequest) -> dict:
    if SESSION_STATE["status"] == "running" and SESSION_STATE["process"] is not None:
        raise HTTPException(status_code=409, detail="A monitoring session is already active.")

    bot_token = _resolve_telegram_token(payload.bot_token)
    chat_id = _resolve_telegram_chat_id(payload.chat_id)

    SESSION_STATE["status"] = "starting"
    SESSION_STATE["started_at"] = None
    SESSION_STATE["logs"] = [
        f"Starting live proctor session on camera index {payload.camera_index}...",
    ]

    env = os.environ.copy()
    if bot_token:
        env["TELEGRAM_BOT_TOKEN"] = bot_token
    if chat_id:
        env["TELEGRAM_CHAT_ID"] = chat_id

    command = [
        sys.executable,
        str(BASE_DIR / "run_camera.py"),
        "--camera",
        str(payload.camera_index),
        "--backend",
        "auto",
    ]

    if not bot_token or not chat_id:
        command.append("--no-telegram")

    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    SESSION_STATE["process"] = process
    SESSION_STATE["pid"] = process.pid
    SESSION_STATE["status"] = "running"
    SESSION_STATE["started_at"] = "live"
    add_log(f"Process started successfully with PID {process.pid}.")

    thread = threading.Thread(target=read_process_output, args=(process,), daemon=True)
    thread.start()

    return {
        "status": "running",
        "pid": process.pid,
        "message": "Live proctor session started.",
    }


@app.post("/api/stop-check")
def stop_live_check() -> dict:
    process = SESSION_STATE.get("process")
    if process is None or process.poll() is not None:
        SESSION_STATE["status"] = "idle"
        SESSION_STATE["pid"] = None
        SESSION_STATE["process"] = None
        return {"status": "idle", "message": "No active session to stop."}

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)

    SESSION_STATE["status"] = "idle"
    SESSION_STATE["pid"] = None
    SESSION_STATE["process"] = None
    add_log("Live session stopped by the user.")
    return {"status": "idle", "message": "Live session stopped."}


@app.post("/analyze_video")
def analyze_video(video_url: str = None):
    video_source = 0 if video_url is None else video_url
    track_eye(video_source)
    detect_head_pose(video_source)
    mouth_opening_detector(video_source)
    detect_phone_and_person(video_source)
    detect_spoofing(video_source)
    return {"message": "Success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)































































































































































































































































"path":"main.py","repo":"MayanKiz/Proctor_AI_V2"}]} oneOfType별로 unique; if not, provide all files as objects? same payload issue. Let's fix with proper tool call. Need be careful; previous tool call weird because I attempted to stuff path in content. Let's do one file at a time or push_files with valid JSON structure. We'll do `push_files` with 3 file objects valid. Need careful no random garbage. Let's also maybe update main.py separately with create_or_update_file to avoid large weird. Simpler to use `create_or_update_file` for main.py and new files with `push_files` for new ones or create_or_update_file per file. We'll do `push_files` with correct structure for web/index.html and web/style.css and update main.py via create_or_update_file after no content issue. Let's first create HTML/CSS. Then update main.py using create_or_update_file. Need maybe easier to do as separate tool calls. Let's do create_or_update_file for HTML and CSS maybe new files. Let's ensure not too long but valid. Need preserve repo path with each file. Let's proceed. Use `create_or_update_file` for main.py with full content replacing existing file. Need maybe use SHA because file exists. We have BlobSha 7b448... from getfile. We can supply sha to update. Good. Let's do main.py first. Need exact code, no weird extra strings. Let's craft carefully. We'll use `create_or_update_file` with `sha` valid. Need also `branch`: 