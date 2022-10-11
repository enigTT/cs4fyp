import cv2 as cv
from src.detectors import detect
from MPModels import pose, drawing


def draw(image, results):
    drawing.draw_landmarks(image, results.pose_landmarks, pose.POSE_CONNECTIONS,
                           drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                           drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))


def start_pose():
    # capture video feed
    cap = cv.VideoCapture(0)

    while True:
        # specifies level of detection and tracking confidence
        with pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as p:
            # get a frame from capture
            val, fr = cap.read()
            if not val:
                exit("Failed to obtain frame")

            # flip image horizontal
            fr = cv.flip(fr, 1)

            img, res = detect(fr, p)

            # draw pose with joints
            draw(img, res)

            # display frame in a window
            cv.imshow("Webacm Image", img)

            # exit on esc keydown
            if cv.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv.destroyAllWindows()
