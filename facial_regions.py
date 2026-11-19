import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE FACIAL REGIONS
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
print("MEDIAPIPE FACIAL REGION DETECTION")
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

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 33

    # Detect landmarks
    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ----------------------------------------------------
        # Draw all landmarks
        # ----------------------------------------------------

        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # Face bounding box
        # ----------------------------------------------------

        x_coords = [
            int(l.x * w)
            for l in landmarks
        ]

        y_coords = [
            int(l.y * h)
            for l in landmarks
        ]

        x_min = max(0, min(x_coords))
        x_max = min(w - 1, max(x_coords))

        y_min = max(0, min(y_coords))
        y_max = min(h - 1, max(y_coords))

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (255, 0, 0),
            2
        )


        # ----------------------------------------------------
        # Highlight important facial areas
        # ----------------------------------------------------

        # Approximate regions based on MediaPipe landmark
        # indices.


        # LEFT EYE
        left_eye = [
            33, 133, 159, 145, 160, 144,
            158, 153
        ]

        # RIGHT EYE
        right_eye = [
            362, 263, 386, 374, 387, 373,
            385, 380
        ]

        # LEFT EYEBROW
        left_eyebrow = [
            70, 63, 105, 66, 107
        ]

        # RIGHT EYEBROW
        right_eyebrow = [
            336, 296, 334, 293, 300
        ]

        # NOSE
        nose = [
            1, 2, 4, 5, 6, 19,
            94, 97, 98, 168
        ]

        # LIPS
        lips = [
            61, 146, 91, 181, 84, 17,
            314, 405, 321, 375, 291,
            308, 324, 318, 402, 317
        ]


        # ----------------------------------------------------
        # Function to highlight landmarks
        # ----------------------------------------------------

        def draw_region(indices, radius=5):

            for index in indices:

                if index < len(landmarks):

                    landmark = landmarks[index]

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        radius,
                        (0, 0, 255),
                        -1
                    )


        # Draw regions
        draw_region(left_eye)
        draw_region(right_eye)
        draw_region(left_eyebrow)
        draw_region(right_eyebrow)
        draw_region(nose)
        draw_region(lips)


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
            f"Landmarks: {len(landmarks)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Red = Facial Regions",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
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
        "MediaPipe Facial Regions",
        frame
    )


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
print("MEDIAPIPE FACIAL REGION DETECTION STOPPED")
print("=" * 60)