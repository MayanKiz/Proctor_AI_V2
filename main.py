# Author: Mynk
from fastapi import FastAPI, APIRouter


from eye_tracker import track_eye
from head_pose_estimation import detect_head_pose
from mouth_opening_detector import mouth_opening_detector
from person_and_phone import detect_phone_and_person


app = FastAPI(title="Mayank Proctoring AI")

MAYANK_BANNER = r"""
███╗   ███╗ █████╗ ██╗   ██╗ █████╗ ███╗   ██╗██╗  ██╗
████╗ ████║██╔══██╗╚██╗ ██╔╝██╔══██╗████╗  ██║██║ ██╔╝
██╔████╔██║███████║ ╚████╔╝ ███████║██╔██╗ ██║█████╔╝
██║╚██╔╝██║██╔══██║  ╚██╔╝  ██╔══██║██║╚██╗██║██╔═██╗
██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║  ██╗
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
"""

@app.on_event("startup")
def show_startup_banner():
    print(MAYANK_BANNER, flush=True)


@app.post("/analyze_video")
def read_root(video_url: str=None):
    track_eye(video_url)
    detect_head_pose(video_url)
    mouth_opening_detector(video_url)
    detect_phone_and_person(video_url)


    return {"message": "Success"}
