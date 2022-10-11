import cv2 as cv
from src.detectors import detect
from MPModels import hands, drawing
from src.trackers.hands import track
from src.controllers.mouse import init


def draw(image, results):
    height, width, _ = image.shape
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            drawing.draw_landmarks(image, landmark, hands.HAND_CONNECTIONS, drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        right = results.multi_hand_landmarks[0]
        points = track(image, right, width, height)
        init(points, image, [width, height])


def start_hands():
    cap = cv.VideoCapture(0)

    with hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6) as h:
        while True:

            val, fr = cap.read()
            if not val:
                exit("Failed to obtain frame")

            fr = cv.flip(fr, 1)

            img, res = detect(fr, h)

            draw(img, res)

            cv.imshow('Virtual Mouse', img)

            # Break gracefully
            if cv.waitKey(10) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()
