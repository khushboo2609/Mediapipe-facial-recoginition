import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"models\face_detector.tflite"


# ============================================================
# CREATE FACE DETECTOR
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.5
)

detector = vision.FaceDetector.create_from_options(options)


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Unable to open webcam.")
    detector.close()
    exit()

print("=" * 60)
print("MEDIAPIPE FACE DETECTION")
print("=" * 60)
print("Model        : BlazeFace")
print("Confidence   : 0.5")
print("Press Q to quit.")
print("=" * 60)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # OpenCV uses BGR.
    # MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect faces
    detection_result = detector.detect(mp_image)

    # Number of detected faces
    face_count = len(detection_result.detections)

    if face_count > 0:

        print(f"Detected {face_count} face(s)")

        for detection in detection_result.detections:

            # Bounding box
            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            # Keep coordinates inside image
            x = max(0, x)
            y = max(0, y)

            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Detection confidence
            score = detection.categories[0].score

            label = f"Face: {score:.2f}"

            cv2.putText(
                frame,
                label,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    else:

        print("No face detected")

    # Display webcam
    cv2.imshow(
        "MediaPipe Face Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
detector.close()
cv2.destroyAllWindows()

print("=" * 60)
print("MEDIAPIPE FACE DETECTION STOPPED")
print("=" * 60)