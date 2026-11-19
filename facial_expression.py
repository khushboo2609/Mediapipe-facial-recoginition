import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MEDIAPIPE FACIAL EXPRESSION / BLENDSHAPE DETECTION
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

    # Enable face blendshapes
    output_face_blendshapes=True,

    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")

    landmarker.close()

    exit()


print("=" * 60)
print("MEDIAPIPE FACIAL EXPRESSION / BLENDSHAPE DETECTION")
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
    # CONVERT BGR → RGB
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
    # FACE LANDMARK + BLENDSHAPE DETECTION
    # ========================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )


    # ========================================================
    # FACE FOUND
    # ========================================================

    if result.face_blendshapes:

        # First detected face
        blendshapes = result.face_blendshapes[0]


        # ----------------------------------------------------
        # Sort blendshapes by score
        # ----------------------------------------------------

        sorted_blendshapes = sorted(
            blendshapes,
            key=lambda x: x.score,
            reverse=True
        )


        # ----------------------------------------------------
        # Display top 10 blendshapes
        # ----------------------------------------------------

        y_position = 50

        cv2.putText(
            frame,
            "TOP FACIAL BLENDSHAPES",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        for category in sorted_blendshapes[:10]:

            name = category.category_name
            score = category.score


            text = f"{name}: {score:.2f}"


            cv2.putText(
                frame,
                text,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1
            )


            y_position += 28


        # ====================================================
        # FIND SPECIFIC EXPRESSIONS
        # ====================================================

        smile_left = 0.0
        smile_right = 0.0

        mouth_open = 0.0

        blink_left = 0.0
        blink_right = 0.0

        brow_down_left = 0.0
        brow_down_right = 0.0


        for category in blendshapes:

            name = category.category_name
            score = category.score


            if name == "mouthSmileLeft":

                smile_left = score


            elif name == "mouthSmileRight":

                smile_right = score


            elif name == "jawOpen":

                mouth_open = score


            elif name == "eyeBlinkLeft":

                blink_left = score


            elif name == "eyeBlinkRight":

                blink_right = score


            elif name == "browDownLeft":

                brow_down_left = score


            elif name == "browDownRight":

                brow_down_right = score


        # ====================================================
        # EXPRESSION SCORES
        # ====================================================

        smile_score = (
            smile_left + smile_right
        ) / 2


        blink_score = (
            blink_left + blink_right
        ) / 2


        brow_down_score = (
            brow_down_left +
            brow_down_right
        ) / 2


        # ====================================================
        # SIMPLE EXPRESSION CLASSIFICATION
        # ====================================================

        if smile_score > 0.50:

            expression = "SMILING"

        elif mouth_open > 0.60:

            expression = "MOUTH OPEN"

        elif brow_down_score > 0.50:

            expression = "BROW LOWERED"

        elif blink_score > 0.50:

            expression = "EYES CLOSED"

        else:

            expression = "NEUTRAL"


        # ====================================================
        # DISPLAY EXPRESSION
        # ====================================================

        cv2.putText(
            frame,
            f"Expression: {expression}",
            (20, h - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3
        )


        # ====================================================
        # DISPLAY IMPORTANT SCORES
        # ====================================================

        cv2.putText(
            frame,
            f"Smile: {smile_score:.2f}",
            (20, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Jaw Open: {mouth_open:.2f}",
            (220, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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


    # ========================================================
    # DISPLAY WINDOW
    # ========================================================

    cv2.imshow(
        "MediaPipe Facial Expression Detection",
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
print("MEDIAPIPE FACIAL EXPRESSION DETECTION STOPPED")
print("=" * 60)