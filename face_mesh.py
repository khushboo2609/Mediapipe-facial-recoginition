import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE FACE MESH VISUALIZATION
# Compatible with MediaPipe 1.0.1
# ============================================================

MODEL_PATH = r"models\face_landmarker.task"


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
# FACE CONNECTION GROUPS
# ============================================================

connections = vision.FaceLandmarksConnections


connection_groups = [
    connections.FACE_LANDMARKS_LIPS,
    connections.FACE_LANDMARKS_LEFT_EYE,
    connections.FACE_LANDMARKS_RIGHT_EYE,
    connections.FACE_LANDMARKS_LEFT_EYEBROW,
    connections.FACE_LANDMARKS_RIGHT_EYEBROW,
    connections.FACE_LANDMARKS_FACE_OVAL,
    connections.FACE_LANDMARKS_NOSE
]


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")

    landmarker.close()

    exit()


print("=" * 60)
print("MEDIAPIPE FACE MESH VISUALIZATION")
print("=" * 60)

print("Webcam opened successfully.")
print("Press Q to quit.")

print("=" * 60)


# ============================================================
# TIMESTAMP
# ============================================================

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


    # ========================================================
    # BGR → RGB
    # ========================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp += 33


    # ========================================================
    # FACE LANDMARK DETECTION
    # ========================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # FACE FOUND
    # ========================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # ====================================================
        # DRAW ALL LANDMARK POINTS
        # ====================================================

        for landmark in landmarks:

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


        # ====================================================
        # DRAW MESH CONNECTIONS
        # ====================================================

        for group in connection_groups:

            for connection in group:

                start_index = connection.start
                end_index = connection.end

                start = landmarks[start_index]
                end = landmarks[end_index]

                x1 = int(start.x * w)
                y1 = int(start.y * h)

                x2 = int(end.x * w)
                y2 = int(end.y * h)

                if (
                    0 <= x1 < w and
                    0 <= y1 < h and
                    0 <= x2 < w and
                    0 <= y2 < h
                ):

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        1
                    )


        # ====================================================
        # DISPLAY LANDMARK COUNT
        # ====================================================

        cv2.putText(
            frame,
            f"Landmarks: {len(landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY STATUS
        # ====================================================

        cv2.putText(
            frame,
            "FACE MESH DETECTED",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
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


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "MediaPipe Face Mesh",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

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
print("MEDIAPIPE FACE MESH VISUALIZATION STOPPED")
print("=" * 60)