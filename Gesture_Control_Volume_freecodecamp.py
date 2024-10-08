import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctype import cast ,   POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities , IAudioEndpointVolume
###########################################################################################
wCam , hCam = 640,480
###########################################################################################

cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
pTime = 0
detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface =  devices.Activate(
    IAudioEndpointVolume._iid_ , CLSCTX_ALL , None
)
volume = cast(interface,POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
minVol = volumeRange[0]
maxVol = volumeRange[1]
vol = 0
volBar = 400
while True:
    success , img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        # print(lmList[4],lmList[8])

        x1 , y1 = lmList[4][1] , lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx , cy = int((x1+x2)//2) , int((y1+y2)//2)

        cv.circle(img,(x1,y1),10,(255,0,255),cv.FILLED)
        cv.circle(img, (x2, y2), 10, (255,0,255), cv.FILLED)
        cv.circle(img,(cx,cy),10,(255,0,255),cv.FILLED)
        cv.line(img,(x1,y1),(x2,y2),(0,255,0),thickness=2)

        lenght = math.hypot(x2-x1,y2-y1)
        # print(lenght)

        # Hand Range 50-300
        # Volume Range -65 to 0

        vol = np.interp(lenght,[50,300],[minVol,maxVol])
        volBar = np.interp(lenght, [50, 300], [400, 150])
        print(lenght, vol)
        volume.SetMasterVolumeLevel(vol, None)

        if lenght< 50 :
            cv.circle(img, (cx, cy), 10, (0, 255, 0), cv.FILLED)

    cv.rectangle(img,(50,150),(85,400),(0,255,0),3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), -1)
    # cv.putText(img, f'FPS:{str(int(fps))}', (10, 50), cv.FONT_HERSHEY_COMPLEX, 1.1, (0, 0, 255), thickness=2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,f'FPS:{str(int(fps))}',(10,50),cv.FONT_HERSHEY_COMPLEX,1.1,(0,0,255),thickness=2)
    cv.imshow("Image",img)
    if cv.waitKey(20) & 0xff == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
