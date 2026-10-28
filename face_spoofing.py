# Author: Mynk
import cv2
import joblib
import numpy as np
import sys

from face_detector import find_faces, get_face_detector


face_model = get_face_detector()
clf = None


def _load_legacy_classifier():
    """Load the old sklearn pickle using current sklearn module locations."""
    import sklearn.ensemble._forest as current_forest
    import sklearn.tree._classes as current_tree
    import joblib.numpy_pickle as current_numpy_pickle

    aliases = {
        "sklearn.ensemble.forest": current_forest,
        "sklearn.tree.tree": current_tree,
        "sklearn.externals.joblib.numpy_pickle": current_numpy_pickle,
    }
    for old_name, current_module in aliases.items():
        sys.modules.setdefault(old_name, current_module)
    return joblib.load("models/face_spoofing.pkl")


def calc_hist(img):
    """Build normalized per-channel histograms for the spoofing classifier."""
    histogram = []
    for channel in range(3):
        values = cv2.calcHist([img], [channel], None, [256], [0, 256])
        maximum = float(values.max())
        if maximum:
            values *= 255.0 / maximum
        histogram.append(values)
    return np.asarray(histogram)


def detect_spoofing(video_source=0):
    """Run the existing face-spoofing classifier on a camera or video source."""
    global clf
    if clf is None:
        clf = _load_legacy_classifier()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source!r}")

    try:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            faces = find_faces(img, face_model)
            for x, y, x1, y1 in faces:
                roi = img[y:y1, x:x1]
                if roi.size == 0:
                    continue

                ycrcb_hist = calc_hist(cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB))
                luv_hist = calc_hist(cv2.cvtColor(roi, cv2.COLOR_BGR2LUV))
                features = np.concatenate((ycrcb_hist.ravel(), luv_hist.ravel()))
                probability = float(clf.predict_proba(features.reshape(1, -1))[0][1])
                label = "Possible spoof" if probability >= 0.7 else "Face appears real"
                color = (0, 0, 255) if probability >= 0.7 else (0, 255, 0)
                cv2.rectangle(img, (x, y), (x1, y1), color, 2)
                cv2.putText(
                    img,
                    f"{label} ({probability:.2f})",
                    (x, max(25, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Face spoofing - press q to continue", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
