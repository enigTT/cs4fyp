import math

import cv2 as cv
import numpy as np
import pyautogui

LMS = None
IMG = None
FRAMER = 120
SMOOTH = 3
SCW, SCH = pyautogui.size()
PT = 0
PLOCX, PLOCY = 0, 0
CLOCX, CLOCY = 0, 0


def init(lms, img, frd):
    global LMS, IMG
    LMS, IMG = lms, img
    peaked = peaks()
    ups = raised()
    handle(frd, ups, peaked)  # Call handle and pass it the frame dimensions and fingers that are up


# doc: Function cheks landmarks for rased poiting and index fingers. These fingers are then marked
def peaks():
    peak = 900
    index = -1

    for lm in LMS:
        if lm[2] < peak:
            peak = lm[2]
            index = lm[0]

    if index == 12:
        cv.circle(IMG, (LMS[12][1], LMS[12][2]), 6, (255, 0, 255), cv.FILLED)
        cv.circle(IMG, (LMS[8][1], LMS[8][2]), 6, (255, 0, 255), cv.FILLED)
    elif index == 8:
        cv.circle(IMG, (LMS[8][1], LMS[8][2]), 6, (0, 255, 255), cv.FILLED)

    return index == 12


def raised():
    fingers = []

    # Checks if thumb is not bent
    if LMS[4][1] < LMS[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Checks if fingers are not bent
    for i in range(8, 21, 4):
        if LMS[i][2] < LMS[i - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def distance(p1, p2, t=1):
    x1, y1 = LMS[p1][1:]
    x2, y2 = LMS[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    cv.line(IMG, (x1, y1), (x2, y2), (0, 255, 255), t)

    length = math.hypot(x2 - x1, y2 - y1)

    return length, [x1, y1, x2, y2, cx, cy]


def handle(frd, fingers, pfing):
    global PLOCY, PLOCX, CLOCY, CLOCX

    cv.rectangle(IMG, (FRAMER, FRAMER), (frd[0] - FRAMER, frd[1] - FRAMER), (255, 0, 255), 2)

    # Image to Screen size ration conversion
    ix = np.interp(LMS[8][1], (FRAMER, frd[0] - FRAMER), (0, SCW))
    iy = np.interp(LMS[8][2], (FRAMER, frd[1] - FRAMER), (0, SCH))

    if fingers[1] == 1 and fingers[2] == 0:
        CLOCX = PLOCX + (ix - PLOCX) // SMOOTH
        CLOCY = PLOCY + (iy - PLOCY) // SMOOTH

        cv.circle(IMG, (LMS[8][1], LMS[8][2]), 6, (255, 255, 0), cv.FILLED)
        pyautogui.moveTo(CLOCX, CLOCY)
        PLOCX, PLOCY = CLOCX, CLOCY

    if fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
        if LMS[0][1] <= 60:
            cv.circle(IMG, (LMS[0][1], LMS[0][2]), 15, (0, 255, 0), cv.FILLED)
            pyautogui.press("pageup")
            pyautogui.sleep(1)
        elif LMS[0][1] >= 580:
            cv.circle(IMG, (LMS[0][1], LMS[0][2]), 15, (0, 255, 0), cv.FILLED)
            pyautogui.press("pagedown")
            pyautogui.sleep(1)

    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
        cv.circle(IMG, (LMS[8][1], LMS[8][2]), 6, (0, 255, 255), cv.FILLED)
        cv.circle(IMG, (LMS[12][1], LMS[12][2]), 6, (0, 255, 255), cv.FILLED)

        pyautogui.sleep(0.25)

        length, cors = distance(8, 12)

        if length < 32 and pfing:

            if fingers[0] == 0:
                pyautogui.doubleClick()
                cv.circle(IMG, (cors[4], cors[5]), 6, (0, 255, 0), cv.FILLED)
                pyautogui.sleep(0.5)
            else:
                pyautogui.click()
                cv.circle(IMG, (cors[4], cors[5]), 6, (0, 255, 0), cv.FILLED)
                pyautogui.sleep(0.5)
