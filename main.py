# Author: Mynk
from fastapi import FastAPI, APIRouter


from eye_tracker import track_eye
from head_pose_estimation import detect_head_pose
from mouth_opening_detector import mouth_opening_detector
from person_and_phone import detect_phone_and_person
from face_spoofing import detect_spoofing


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
    video_source = 0 if video_url is None else video_url
    track_eye(video_source)
    detect_head_pose(video_source)
    mouth_opening_detector(video_source)
    detect_phone_and_person(video_source)
    detect_spoofing(video_source)


    return {"message": "Success"}
