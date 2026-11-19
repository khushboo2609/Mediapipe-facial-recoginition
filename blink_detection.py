import cv2
import math
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE BLINK DETECTION
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
# EYE LANDMARK INDICES
# ============================================================

# Left eye
LEFT_EYE = [
    33,   # outer corner
    133,  # inner corner
    159,  # upper eyelid
    145   # lower eyelid
]

# Right eye
RIGHT_EYE = [
    362,  # inner corner
    263,  # outer corner
    386,  # upper eyelid
    374   # lower eyelid
]


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def calculate_ear(landmarks, eye_indices):

    outer = landmarks[eye_indices[0]]
    inner = landmarks[eye_indices[1]]

    upper = landmarks[eye_indices[2]]
    lower = landmarks[eye_indices[3]]

    horizontal_distance = distance(
        outer,
        inner
    )

    vertical_distance = distance(
        upper,
        lower
    )

    if horizontal_distance == 0:
        return 0

    ear = vertical_distance / horizontal_distance

    return ear


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")

    landmarker.close()

    exit()


print("=" * 60)
print("MEDIAPIPE EYE BLINK DETECTION")
print("=" * 60)

print("Webcam opened successfully.")
print("Press Q to quit.")

print("=" * 60)


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0

blink_count = 0

eyes_closed = False

# Number of consecutive frames required
# before considering the eyes closed
closed_frame_count = 0

CLOSED_FRAMES_REQUIRED = 3

EAR_THRESHOLD = 0.20


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
        # Calculate EAR
        # ----------------------------------------------------

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )


        # Average both eyes

        average_ear = (
            left_ear + right_ear
        ) / 2


        # ----------------------------------------------------
        # BLINK DETECTION
        # ----------------------------------------------------

        if average_ear < EAR_THRESHOLD:
            closed_frame_count += 1

            cv2.putText(
            frame,
            "EYES CLOSING",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
            )

            if (
            closed_frame_count >= CLOSED_FRAMES_REQUIRED
            and not eyes_closed
            ):
                blink_count += 1
                eyes_closed = True

        else:
            closed_frame_count = 0
            eyes_closed = False

            cv2.putText(
                frame,
                "EYES OPEN",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
                )
        # ----------------------------------------------------
        # DISPLAY EAR
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"EAR: {average_ear:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # DISPLAY BLINK COUNT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Blinks: {blink_count}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


        # ----------------------------------------------------
        # DRAW EYE LANDMARKS
        # ----------------------------------------------------

        eye_points = LEFT_EYE + RIGHT_EYE

        for index in eye_points:

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
        "MediaPipe Eye Blink Detection",
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
print("MEDIAPIPE EYE BLINK DETECTION STOPPED")
print(f"Total blinks detected: {blink_count}")
print("=" * 60)