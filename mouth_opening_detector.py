# -*- coding: utf-8 -*-
"""
@author: Mynk
"""

import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

face_model = get_face_detector()
landmark_model = get_landmark_model()
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
inner_points = [[61, 67], [62, 66], [63, 65]]
font = cv2.FONT_HERSHEY_SIMPLEX


def mouth_opening_detector(video_path=0):
    """Calibrate a neutral mouth pose, then report mouth opening from a video source."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_path!r}")

    baseline_outer = [0.0] * len(outer_points)
    baseline_inner = [0.0] * len(inner_points)
    calibrated = False
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            rects = find_faces(img, face_model)
            shape = None
            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                draw_marks(img, shape)
                break

            message = "Face detected - press r to calibrate" if shape is not None else "Position face in camera"
            cv2.putText(img, message, (30, 30), font, 0.8, (0, 255, 255), 2)
            cv2.imshow("Mouth opening - press r to calibrate", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return
            if key != ord("r"):
                continue
            if shape is None:
                print("No face found for calibration; face the camera and press r again.")
                continue

            sample_count = 0
            for _ in range(100):
                ret, sample = cap.read()
                if not ret:
                    break
                sample_faces = find_faces(sample, face_model)
                if not sample_faces:
                    continue
                sample_shape = detect_marks(sample, landmark_model, sample_faces[0])
                for i, (p1, p2) in enumerate(outer_points):
                    baseline_outer[i] += sample_shape[p2][1] - sample_shape[p1][1]
                for i, (p1, p2) in enumerate(inner_points):
                    baseline_inner[i] += sample_shape[p2][1] - sample_shape[p1][1]
                sample_count += 1

            if sample_count == 0:
                print("Calibration failed: no face frames were captured.")
                continue
            baseline_outer = [value / sample_count for value in baseline_outer]
            baseline_inner = [value / sample_count for value in baseline_inner]
            calibrated = True
            print("Mouth calibration complete. Press q to finish this check.")
            break

        if not calibrated:
            return

        while True:
            ret, img = cap.read()
            if not ret:
                break
            rects = find_faces(img, face_model)
            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                outer_open = sum(
                    baseline_outer[i] + 3 < shape[p2][1] - shape[p1][1]
                    for i, (p1, p2) in enumerate(outer_points)
                )
                inner_open = sum(
                    baseline_inner[i] + 2 < shape[p2][1] - shape[p1][1]
                    for i, (p1, p2) in enumerate(inner_points)
                )
                draw_marks(img, shape[48:])
                if outer_open > 3 and inner_open > 2:
                    print("Mouth open")
                    cv2.putText(img, "Mouth open", (30, 30), font, 1, (0, 255, 255), 2)

            cv2.imshow("Mouth opening - press q to continue", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
