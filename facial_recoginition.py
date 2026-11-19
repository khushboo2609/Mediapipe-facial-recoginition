import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE FACE RECOGNITION
# ============================================================

print("=" * 60)
print("MEDIAPIPE FACE RECOGNITION")
print("=" * 60)


# ------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------

MODEL_PATH = "models/face_landmarker.task"

if not os.path.exists(MODEL_PATH):

    print("ERROR: Face landmarker model not found.")
    print("Expected:", MODEL_PATH)
    exit()


# ------------------------------------------------------------
# MEDIAPIPE FACE LANDMARKER
# ------------------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_features(landmarks):

    points = []

    for landmark in landmarks:

        points.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    points = np.array(
        points,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # NOSE AS REFERENCE POINT
    # --------------------------------------------------------

    nose = points[1].copy()

    points = points - nose

    # --------------------------------------------------------
    # SCALE NORMALIZATION
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points,
        axis=1
    )

    max_distance = np.max(distances)

    if max_distance > 0:

        points = points / max_distance

    return points.flatten()


# ------------------------------------------------------------
# COSINE SIMILARITY
# ------------------------------------------------------------

def calculate_similarity(
    reference,
    current
):

    reference_norm = np.linalg.norm(
        reference
    )

    current_norm = np.linalg.norm(
        current
    )

    if reference_norm == 0 or current_norm == 0:

        return 0.0

    similarity = np.dot(
        reference,
        current
    ) / (
        reference_norm * current_norm
    )

    return float(similarity)


# ------------------------------------------------------------
# WEBCAM
# ------------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened successfully.")
print("Look directly at the camera.")
print("Press SPACE to capture your reference face.")
print("Press R to recapture your reference face.")
print("Press Q to quit.")
print("=" * 60)


# ------------------------------------------------------------
# VARIABLES
# ------------------------------------------------------------

timestamp = 0

reference_features = None

reference_captured = False

similarity_score = 0.0

RECOGNITION_THRESHOLD = 0.90


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read webcam frame.")
        break


    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MEDIAPIPE IMAGE
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp += 33


    # --------------------------------------------------------
    # FACE LANDMARK DETECTION
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # --------------------------------------------------------
    # FACE DETECTED
    # --------------------------------------------------------

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        current_features = extract_features(
            landmarks
        )


        # ----------------------------------------------------
        # RECOGNITION
        # ----------------------------------------------------

        if reference_captured:

            similarity_score = calculate_similarity(
                reference_features,
                current_features
            )

            if similarity_score >= RECOGNITION_THRESHOLD:

                label = "MATCH: Khushboo"

            else:

                label = "UNKNOWN"


        else:

            label = "Press SPACE to capture"


        # ----------------------------------------------------
        # DISPLAY LABEL
        # ----------------------------------------------------

        cv2.putText(
            frame,
            label,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # DISPLAY SIMILARITY
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Similarity: {similarity_score:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


        # ----------------------------------------------------
        # DISPLAY LANDMARK COUNT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Landmarks: {len(landmarks)}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


    # --------------------------------------------------------
    # SHOW WINDOW
    # --------------------------------------------------------

    cv2.imshow(
        "MediaPipe Face Recognition",
        frame
    )


    # --------------------------------------------------------
    # KEYBOARD INPUT
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # SPACE → CAPTURE REFERENCE
    # --------------------------------------------------------

    if key == ord(' '):

        if result.face_landmarks:

            reference_features = current_features.copy()

            reference_captured = True

            similarity_score = 1.0

            print("=" * 60)
            print("REFERENCE FACE CAPTURED")
            print("Label: Khushboo")
            print("Landmarks:", len(landmarks))
            print("=" * 60)

        else:

            print("No face detected. Try again.")


    # --------------------------------------------------------
    # R → RESET / RECAPTURE
    # --------------------------------------------------------

    elif key == ord('r'):

        reference_features = None

        reference_captured = False

        similarity_score = 0.0

        print("Reference face cleared.")
        print("Press SPACE to capture again.")


    # --------------------------------------------------------
    # Q → QUIT
    # --------------------------------------------------------

    elif key == ord('q'):

        print("Q pressed.")
        break


# ------------------------------------------------------------
# CLEANUP
# ------------------------------------------------------------

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print("=" * 60)
print("MEDIAPIPE FACE RECOGNITION STOPPED")
print("=" * 60)