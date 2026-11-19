import cv2
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE MOUTH OPEN DETECTION
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

landmarker = vision.FaceLandmarker.create_from_options(options)


# ============================================================
# MOUTH LANDMARKS
# ============================================================

# Upper and lower inner lip points
UPPER_LIP = 13
LOWER_LIP = 14

# Left and right mouth corners
LEFT_CORNER = 61
RIGHT_CORNER = 291


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ============================================================
# MOUTH OPENING RATIO
# ============================================================

def calculate_mouth_ratio(landmarks):

    upper_lip = landmarks[UPPER_LIP]
    lower_lip = landmarks[LOWER_LIP]

    left_corner = landmarks[LEFT_CORNER]
    right_corner = landmarks[RIGHT_CORNER]

    vertical_distance = distance(
        upper_lip,
        lower_lip
    )

    horizontal_distance = distance(
        left_corner,
        right_corner
    )

    if horizontal_distance == 0:
        return 0

    mouth_ratio = (
        vertical_distance /
        horizontal_distance
    )

    return mouth_ratio


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")

    landmarker.close()

    exit()


print("=" * 60)
print("MEDIAPIPE MOUTH OPEN DETECTION")
print("=" * 60)

print("Webcam opened successfully.")
print("Press Q to quit.")

print("=" * 60)


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0

mouth_open_count = 0

mouth_is_open = False

# Starting threshold
MOUTH_THRESHOLD = 0.30


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
    # CONVERT IMAGE
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


        # ----------------------------------------------------
        # Calculate mouth ratio
        # ----------------------------------------------------

        mouth_ratio = calculate_mouth_ratio(
            landmarks
        )


        # ----------------------------------------------------
        # MOUTH DETECTION
        # ----------------------------------------------------

        if mouth_ratio > MOUTH_THRESHOLD:

            cv2.putText(
                frame,
                "MOUTH OPEN",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )


            if not mouth_is_open:

                mouth_open_count += 1

                mouth_is_open = True


        else:

            cv2.putText(
                frame,
                "MOUTH CLOSED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            mouth_is_open = False


        # ----------------------------------------------------
        # DISPLAY MOUTH RATIO
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"MOR: {mouth_ratio:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # DISPLAY OPEN COUNT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Mouth Opens: {mouth_open_count}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


        # ----------------------------------------------------
        # DRAW MOUTH LANDMARKS
        # ----------------------------------------------------

        mouth_points = [
            UPPER_LIP,
            LOWER_LIP,
            LEFT_CORNER,
            RIGHT_CORNER
        ]

        for index in mouth_points:

            landmark = landmarks[index]

            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )

            cv2.circle(
                frame,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )


        # ----------------------------------------------------
        # DRAW MOUTH LINE
        # ----------------------------------------------------

        upper = landmarks[UPPER_LIP]
        lower = landmarks[LOWER_LIP]

        x1 = int(upper.x * w)
        y1 = int(upper.y * h)

        x2 = int(lower.x * w)
        y2 = int(lower.y * h)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
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
        "MediaPipe Mouth Detection",
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
print("MEDIAPIPE MOUTH OPEN DETECTION STOPPED")
print(
    f"Total mouth openings detected: {mouth_open_count}"
)
print("=" * 60)