import cv2 as cv
from src.detectors import detect
from MPModels import holistic, drawing


def draw(image, results):
    # Draw pose connections
    drawing.draw_landmarks(image, results.pose_landmarks, holistic.POSE_CONNECTIONS,
                           drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                           drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                           )
    # Draw left hand connections
    drawing.draw_landmarks(image, results.left_hand_landmarks, holistic.HAND_CONNECTIONS,
                           drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                           drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                           )
    # Draw right hand connections
    drawing.draw_landmarks(image, results.right_hand_landmarks, holistic.HAND_CONNECTIONS,
                           drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                           drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                           )


def start_holistic():
    cap = cv.VideoCapture(0)

    with holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as hol:
        while True:

            val, fr = cap.read()
            if not val:
                exit("Failed to obtain frame")

            fr = cv.flip(fr, 1)

            img, res = detect(fr, hol)

            draw(img, res)

            cv.imshow('OpenCV Feed', img)

            if cv.waitKey(10) & 0xFF == 27:
                break

        cap.release()
        cv.destroyAllWindows()
