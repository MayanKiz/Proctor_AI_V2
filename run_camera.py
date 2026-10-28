# Author: Mynk
"""Run all available visual proctoring checks from the local webcam."""

from eye_tracker import track_eye
from face_spoofing import detect_spoofing
from head_pose_estimation import detect_head_pose
from mouth_opening_detector import mouth_opening_detector
from person_and_phone import detect_phone_and_person


BANNER = r"""
███╗   ███╗ █████╗ ██╗   ██╗ █████╗ ███╗   ██╗██╗  ██╗
████╗ ████║██╔══██╗╚██╗ ██╔╝██╔══██╗████╗  ██║██║ ██╔╝
██╔████╔██║███████║ ╚████╔╝ ███████║██╔██╗ ██║█████╔╝
██║╚██╔╝██║██╔══██║  ╚██╔╝  ██╔══██║██║╚██╗██║██╔═██╗
██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║  ██╗
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
"""


FEATURES = [
    ("Eye tracking", track_eye),
    ("Head pose", detect_head_pose),
    ("Mouth opening", mouth_opening_detector),
    ("Person and phone detection", detect_phone_and_person),
    ("Face spoofing", detect_spoofing),
]


def main():
    print(BANNER)
    print("The webcam checks run one after another. Press q in a feature window to continue.")
    print("For mouth opening, press r to calibrate while facing the camera, then q to continue.")
    print("The first person/phone check may download the YOLOv3 weights (~236 MB).\n")

    for title, run_feature in FEATURES:
        print(f"\nStarting {title}. Press q in its OpenCV window when ready for the next check.")
        if title == "Mouth opening":
            print("Press r once to record the initial mouth measurements; then press q to move on.")
        try:
            run_feature(0)
        except Exception as exc:
            print(f"{title} could not run: {type(exc).__name__}: {exc}")
            print("Continuing to the next visual check. Verify the webcam, model files, and dependencies.")

    print("\nAll visual checks have finished.")
    print("Audio monitoring is separate; it needs a microphone and a paper.txt file.")


if __name__ == "__main__":
    main()
