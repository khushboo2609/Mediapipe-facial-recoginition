import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# 1. MODEL PATH
# ============================================================

MODEL_PATH = r"models\face_landmarker.task"


# ============================================================
# 2. CREATE FACE LANDMARKER
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

landmarker = vision.FaceLandmarker.create_from_options(options)


# ============================================================
# 3. START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()

print("=" * 60)
print("MEDIAPIPE FACE LANDMARKER")
print("=" * 60)
print("Model        : Face Landmarker")
print("Faces        : 1")
print("Confidence   : 0.5")
print("Press Q to quit.")
print("=" * 60)

frame_timestamp_ms = 0


# ============================================================
# 4. PROCESS WEBCAM
# ============================================================

while True:

    print("Reading frame...")

    ret, frame = cap.read()

    print("Frame status:", ret)

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Flip image horizontally for a mirror-like webcam view
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Increase timestamp for every frame
    frame_timestamp_ms += 33

    # Run Face Landmarker
    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )
    print("Landmarker processed frame.")
    print("Faces detected:", len(result.face_landmarks))


    # ========================================================
    # 5. DRAW LANDMARKS
    # ========================================================

    if result.face_landmarks:

        h, w, _ = frame.shape

        for face_landmarks in result.face_landmarks:

            for landmark in face_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if 0 <= x < w and 0 <= y < h:

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

        cv2.putText(
            frame,
            "Face Landmarks Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Faces: {len(result.face_landmarks)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No Face Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # 6. DISPLAY
    # ========================================================

    cv2.imshow(
        "MediaPipe Face Landmarker",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 7. CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("=" * 60)
print("MEDIAPIPE FACE LANDMARKER STOPPED")
print("=" * 60)