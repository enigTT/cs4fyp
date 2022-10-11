import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


model = load_model('trained_keras_model')

f = open('gestures.txt', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)



cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    result = hands.process(framergb)

    
    className = ''


    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])


            prediction = model.predict([landmarks])

            classID = np.argmax(prediction)
            className = classNames[classID]


    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)


    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()
