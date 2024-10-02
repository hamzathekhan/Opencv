from time import process_time

import cv2 as cv
import mediapipe as mp
import time


cap = cv.VideoCapture(0)
pTime  = 0
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)
while True:
    success , img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,faceLm,
                                  landmark_drawing_spec=drawSpec,)

            for id , lm in enumerate(faceLm.landmark):
                ih , iw , ic = img.shape
                cx , cy = int(lm.x * iw) , int(lm.y * ih)
                print(id , cx,cy)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),2)
    cv.imshow("Image",img)
    if cv.waitKey(20) & 0xff == ord("q"):
        break

cap.release()
cv.destroyAllWindows()