import cv2 as cv
import mediapipe as mp
import numpy as np


cap = cv.VideoCapture(0)
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    isTrue , img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        LmList = []
        open_fingers = 0
        for single_hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,single_hand)
            for id , lm in enumerate(single_hand.landmark):
                imgH , imgW , imgC = img.shape
                cx , cy = int(lm.x * imgW) , int(lm.y * imgH)
                LmList.append([id,cx,cy])

                # Thumb
                if id == 4 :
                    if len(LmList) != 0:
                        if LmList[4][1] > LmList[2][1]:
                            open_fingers += 1
                        else:
                            open_fingers -= 1
                            if open_fingers < 0:
                                open_fingers = 0
                # Other Fingers
                if id in [8,12,16,20]:
                    if LmList[id][2] < LmList[id-2][2]:
                        open_fingers += 1
                    else:
                        open_fingers -= 1
                        if open_fingers < 0:
                            open_fingers = 0
            print(open_fingers)




    cv.imshow("Image",img)
    if cv.waitKey(20) & 0xff == ord("q"):
        break

cap.release()
cv.destroyAllWindows()