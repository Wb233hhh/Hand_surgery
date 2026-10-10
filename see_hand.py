import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 21 个关键点里，哪些点连成骨头
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

def main():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("打不开摄像头")

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # 镜像，手往左动，画面也往左
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame.shape[:2]
        if result.hand_landmarks:
            pts = result.hand_landmarks[0]
            xy = [(int(p.x * w), int(p.y * h)) for p in pts]

            for a, b in BONES:
                cv2.line(frame, xy[a], xy[b], (80, 220, 120), 2)
            for x, y in xy:
                cv2.circle(frame, (x, y), 4, (40, 180, 255), -1)

            thumb, index = xy[4], xy[8]  # 4 拇指尖，8 食指尖
            cv2.line(frame, thumb, index, (0, 200, 255), 2)
            pinch = ((thumb[0] - index[0]) ** 2 + (thumb[1] - index[1]) ** 2) ** 0.5

            cv2.circle(frame, index, 8, (0, 0, 255), 2)
            cv2.putText(frame, f"pinch {pinch:.0f}px", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
            cv2.putText(frame, f"index ({index[0]}, {index[1]})", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 80), 2)
        else:
            cv2.putText(frame, "no hand", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("see hand", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()