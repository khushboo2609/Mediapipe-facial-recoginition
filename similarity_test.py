import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE FACE RECOGNITION - SIMILARITY TEST
# ============================================================

print("=" * 60)
print("MEDIAPIPE FACE RECOGNITION - SIMILARITY TEST")
print("=" * 60)


MODEL_PATH = "models/face_landmarker.task"


if not os.path.exists(MODEL_PATH):

    print("ERROR: Model not found.")
    print("Expected:", MODEL_PATH)
    exit()


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

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

    # Use nose as reference
    nose = points[1].copy()

    points = points - nose

    # Scale normalization
    distances = np.linalg.norm(
        points,
        axis=1
    )

    max_distance = np.max(distances)

    if max_distance > 0:

        points = points / max_distance

    return points.flatten()


# ============================================================
# COSINE SIMILARITY
# ============================================================

def calculate_similarity(reference, current):

    reference_norm = np.linalg.norm(reference)

    current_norm = np.linalg.norm(current)

    if reference_norm == 0 or current_norm == 0:

        return 0.0

    similarity = np.dot(
        reference,
        current
    ) / (
        reference_norm * current_norm
    )

    return float(similarity)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")
    exit()


print("Webcam opened successfully.")
print()
print("STEP 1:")
print("Look directly at the camera.")
print("Press SPACE to capture your reference face.")
print()
print("STEP 2:")
print("Move your face naturally.")
print("Similarity values will be printed below.")
print()
print("STEP 3:")
print("Test another person if available.")
print()
print("Press R to reset reference.")
print("Press Q to quit.")
print("=" * 60)


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0

reference_features = None

reference_captured = False

sample_counter = 0

similarity_values = []


# Print similarity every N frames
PRINT_EVERY = 10


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read webcam frame.")
        break


    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # RGB CONVERSION
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    timestamp += 33


    # --------------------------------------------------------
    # LANDMARK DETECTION
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
        # SIMILARITY
        # ----------------------------------------------------

        if reference_captured:

            similarity = calculate_similarity(
                reference_features,
                current_features
            )

            similarity_values.append(
                similarity
            )

            sample_counter += 1


            # ------------------------------------------------
            # PRINT TO POWERSHELL
            # ------------------------------------------------

            if sample_counter % PRINT_EVERY == 0:

                print(
                    f"Sample {sample_counter:03d} | "
                    f"Similarity: {similarity:.4f}"
                )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Similarity: {similarity:.3f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                3
            )


        else:

            cv2.putText(
                frame,
                "Press SPACE to capture reference",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


        cv2.putText(
            frame,
            f"Landmarks: {len(landmarks)}",
            (20, 90),
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
    # DISPLAY
    # --------------------------------------------------------

    cv2.imshow(
        "MediaPipe Similarity Testing",
        frame
    )


    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # SPACE - CAPTURE REFERENCE
    # --------------------------------------------------------

    if key == ord(' '):

        if result.face_landmarks:

            reference_features = current_features.copy()

            reference_captured = True

            similarity_values = []

            sample_counter = 0

            print()
            print("=" * 60)
            print("REFERENCE CAPTURED")
            print("Person: Khushboo")
            print("Landmarks:", len(landmarks))
            print("=" * 60)
            print()


        else:

            print("No face detected.")


    # --------------------------------------------------------
    # R - RESET
    # --------------------------------------------------------

    elif key == ord('r'):

        reference_features = None

        reference_captured = False

        similarity_values = []

        sample_counter = 0

        print()
        print("Reference cleared.")
        print("Press SPACE to capture again.")
        print()


    # --------------------------------------------------------
    # Q - QUIT
    # --------------------------------------------------------

    elif key == ord('q'):

        print()
        print("Q pressed.")
        break


# ============================================================
# FINAL STATISTICS
# ============================================================

print()
print("=" * 60)
print("SIMILARITY TEST RESULTS")
print("=" * 60)


if similarity_values:

    values = np.array(
        similarity_values
    )

    print(
        f"Number of samples : {len(values)}"
    )

    print(
        f"Minimum similarity: {np.min(values):.4f}"
    )

    print(
        f"Maximum similarity: {np.max(values):.4f}"
    )

    print(
        f"Average similarity: {np.mean(values):.4f}"
    )

    print(
        f"Standard deviation: {np.std(values):.4f}"
    )

else:

    print("No similarity samples collected.")


print("=" * 60)


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print("MEDIAPIPE SIMILARITY TEST STOPPED")
print("=" * 60)