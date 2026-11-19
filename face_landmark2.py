import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE FACE LANDMARKER
# ============================================================

MODEL_PATH = r"models\face_landmarker.task"

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

landmarker = vision.FaceLandmarker.create_from_options(options)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    landmarker.close()
    exit()

print("=" * 60)
print("MEDIAPIPE FACE LANDMARKER")
print("=" * 60)
print("Webcam opened successfully.")
print("Press Q to quit.")
print("=" * 60)


timestamp = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp
    timestamp += 33

    # Detect landmarks
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # IF FACE DETECTED
    # ========================================================

    if result.face_landmarks:

        face_landmarks = result.face_landmarks[0]

        # ----------------------------------------------------
        # Draw ALL landmarks
        # ----------------------------------------------------

        for landmark in face_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            # Keep coordinates inside image
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # Calculate face bounding box
        # ----------------------------------------------------

        x_coords = [
            int(landmark.x * w)
            for landmark in face_landmarks
        ]

        y_coords = [
            int(landmark.y * h)
            for landmark in face_landmarks
        ]

        x_min = max(0, min(x_coords))
        x_max = min(w - 1, max(x_coords))

        y_min = max(0, min(y_coords))
        y_max = min(h - 1, max(y_coords))


        # ----------------------------------------------------
        # Draw BLUE bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (255, 0, 0),
            2
        )


        # ----------------------------------------------------
        # Display information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Landmarks: {len(face_landmarks)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "MediaPipe Face Landmarker",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        print("Q pressed.")
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("=" * 60)
print("MEDIAPIPE FACE LANDMARKER STOPPED")
print("=" * 60)