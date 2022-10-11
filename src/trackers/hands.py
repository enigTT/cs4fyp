import cv2 as cv
from MPModels import hands, drawing

"""Module Doc
Returns the coordinates of the points within a single palm.
Used within the holistic and hand detectors, right after obtaining the landmarks
Returns an array with the landmark point as index
"""

def track(img, hand, w, h):

    store = []

    # Checks for key points their location in the video frame
    for pid, lm in enumerate(hand.landmark):
        xcor, ycor = round(lm.x * w), round(lm.y * h)
        store.append([pid, xcor, ycor])
        if pid == 0:
            cv.circle(img, (xcor, ycor), 15, (255, 0, 255), cv.FILLED)

    # Returns the array consisting of all points and their current position within the frame
    return store
