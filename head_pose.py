import cv2
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE HEAD POSE DETECTION
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
# IMPORTANT FACE LANDMARKS
# ============================================================

NOSE = 1

LEFT_EYE = 33
RIGHT_EYE = 263

LEFT_EAR = 234
RIGHT_EAR = 454

CHIN = 152

FOREHEAD = 10


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ============================================================
# HEAD POSE CALCULATION
# ============================================================

def calculate_head_pose(landmarks):

    nose = landmarks[NOSE]

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]

    left_ear = landmarks[LEFT_EAR]
    right_ear = landmarks[RIGHT_EAR]

    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]


    # --------------------------------------------------------
    # FACE CENTER
    # --------------------------------------------------------

    face_center_x = (
        left_ear.x + right_ear.x
    ) / 2

    face_center_y = (
        forehead.y + chin.y
    ) / 2


    # --------------------------------------------------------
    # HORIZONTAL HEAD POSITION
    # --------------------------------------------------------

    horizontal_offset = (
        nose.x - face_center_x
    )


    # Distance between ears
    ear_distance = distance(
        left_ear,
        right_ear
    )


    if ear_distance == 0:

        horizontal_ratio = 0

    else:

        horizontal_ratio = (
            horizontal_offset /
            ear_distance
        )


    # --------------------------------------------------------
    # VERTICAL HEAD POSITION
    # --------------------------------------------------------

    face_height = distance(
        forehead,
        chin
    )


    vertical_offset = (
        nose.y - face_center_y
    )


    if face_height == 0:

        vertical_ratio = 0

    else:

        vertical_ratio = (
            vertical_offset /
            face_height
        )


    return horizontal_ratio, vertical_ratio


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")

    landmarker.close()

    exit()


print("=" * 60)
print("MEDIAPIPE HEAD POSE DETECTION")
print("=" * 60)

print("Webcam opened successfully.")
print("Press Q to quit.")

print("=" * 60)


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0


# Starting thresholds
HORIZONTAL_THRESHOLD = 0.08
VERTICAL_THRESHOLD = 0.08


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
        # Calculate head position
        # ----------------------------------------------------

        horizontal_ratio, vertical_ratio = (
            calculate_head_pose(landmarks)
        )


        # ----------------------------------------------------
        # HEAD DIRECTION
        # ----------------------------------------------------

        if horizontal_ratio > HORIZONTAL_THRESHOLD:

            direction = "LOOKING RIGHT"

        elif horizontal_ratio < -HORIZONTAL_THRESHOLD:

            direction = "LOOKING LEFT"

        elif vertical_ratio > VERTICAL_THRESHOLD:

            direction = "LOOKING DOWN"

        elif vertical_ratio < -VERTICAL_THRESHOLD:

            direction = "LOOKING UP"

        else:

            direction = "LOOKING STRAIGHT"


        # ----------------------------------------------------
        # DISPLAY DIRECTION
        # ----------------------------------------------------

        cv2.putText(
            frame,
            direction,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


        # ----------------------------------------------------
        # DISPLAY RATIOS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Horizontal: {horizontal_ratio:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Vertical: {vertical_ratio:.3f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # DRAW IMPORTANT LANDMARKS
        # ----------------------------------------------------

        important_points = [
            NOSE,
            LEFT_EYE,
            RIGHT_EYE,
            LEFT_EAR,
            RIGHT_EAR,
            FOREHEAD,
            CHIN
        ]


        for index in important_points:

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
                5,
                (0, 0, 255),
                -1
            )


        # ----------------------------------------------------
        # DRAW NOSE DIRECTION LINE
        # ----------------------------------------------------

        nose = landmarks[NOSE]

        nose_x = int(
            nose.x * w
        )

        nose_y = int(
            nose.y * h
        )

        line_length = 100

        if horizontal_ratio > 0:

            end_x = nose_x + line_length

        else:

            end_x = nose_x - line_length


        cv2.line(
            frame,
            (nose_x, nose_y),
            (end_x, nose_y),
            (255, 0, 0),
            3
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
        "MediaPipe Head Pose Detection",
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
print("MEDIAPIPE HEAD POSE DETECTION STOPPED")
print("=" * 60)