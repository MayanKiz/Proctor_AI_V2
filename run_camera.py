# Author: Mynk
"""Run V1's full proctoring session, followed by V2's visual checks."""

import argparse
from pathlib import Path

import v1_config
from v1_proctoring import main as run_v1


BANNER = r"""
███╗   ███╗ █████╗ ██╗   ██╗ █████╗ ███╗   ██╗██╗  ██╗
████╗ ████║██╔══██╗╚██╗ ██╔╝██╔══██╗████╗  ██║██║ ██╔╝
██╔████╔██║███████║ ╚████╔╝ ███████║██╔██╗ ██║█████╔╝
██║╚██╔╝██║██╔══██║  ╚██╔╝  ██╔══██║██║╚██╗██║██╔═██╗
██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██║ ╚████║██║  ██╗
╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Run V1 proctoring, then V2 visual tests")
    parser.add_argument("--camera", type=int, default=v1_config.CAMERA_INDEX, help="webcam index (default: 0)")
    parser.add_argument(
        "--backend", choices=["auto", "legacy", "tasks", "haar"],
        default=v1_config.FACE_BACKEND if v1_config.FACE_BACKEND in {"auto", "legacy", "tasks", "haar"} else "auto",
        help="V1 face backend (default: auto)",
    )
    parser.add_argument("--no-telegram", action="store_true", help="never prompt for Telegram; run locally only")
    return parser.parse_args()


def run_v2_checks(camera_index):
    # Import only after V1 ends so the first user's experience follows V1's startup.
    from eye_tracker import track_eye
    from face_spoofing import detect_spoofing
    from head_pose_estimation import detect_head_pose
    from mouth_opening_detector import mouth_opening_detector
    from person_and_phone import detect_phone_and_person

    features = [
        ("Eye tracking", track_eye),
        ("Head pose", detect_head_pose),
        ("Mouth opening", mouth_opening_detector),
        ("Person and phone detection", detect_phone_and_person),
        ("Face spoofing", detect_spoofing),
    ]
    print("\n" + "=" * 72)
    print("V1 session finished. Starting the V2 visual checks now.")
    print("Press q in each V2 window to continue. For mouth calibration, press r.")
    print("The first person/phone check may download YOLOv3 weights (~236 MB).")
    print("=" * 72)

    for title, run_feature in features:
        print(f"\nStarting V2: {title}. Press q in the OpenCV window to continue.")
        if title == "Mouth opening":
            print("Face the camera and press r once to calibrate; then press q after the check.")
        try:
            run_feature(camera_index)
        except Exception as exc:
            print(f"{title} could not run: {type(exc).__name__}: {exc}")
            print("Skipping this V2 check and continuing. Review the error, camera, dependencies, or model files.")


def maybe_run_audio_check():
    paper_path = Path(__file__).resolve().parent / "paper.txt"
    if not paper_path.is_file():
        print("\nV2 audio check skipped: add your exam question text to paper.txt to enable it.")
        return
    print("\nOptional V2 audio check: it records from the microphone and sends that audio to Google Speech Recognition for transcription.")
    print("The transcript is saved locally as test.txt. Do not continue if this data sharing is not acceptable.")
    try:
        consent = input("Type YES to run the audio check, or press Enter to skip: ").strip()
    except EOFError:
        consent = ""
    if consent != "YES":
        print("Audio check skipped; no audio was recorded or sent.")
        return
    from audio_part import run_audio_check
    run_audio_check()


def main():
    args = parse_args()
    print(BANNER)
    print("Combined mode: V1 runs first until you press q; then V2 visual tests run in sequence.")
    print("V1 Telegram alerts are optional; choose Skip, or use --no-telegram for local-only mode.")

    v1_args = ["--camera", str(args.camera), "--backend", args.backend]
    if args.no_telegram:
        v1_config.TELEGRAM_BOT_TOKEN = ""
        v1_config.TELEGRAM_CHAT_ID = ""
        v1_args.append("--non-interactive")
    v1_result = run_v1(v1_args)
    if v1_result != 0:
        print(f"V1 stopped with exit code {v1_result}; V2 will not start because the first stage did not complete.")
        return v1_result

    run_v2_checks(args.camera)
    maybe_run_audio_check()
    print("\nV1 and V2 checks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
