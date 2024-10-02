import cv2 as cv
import mediapipe as mp
import time

from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS

cap = cv.VideoCapture("sdfs.mp4")
pTime = 0
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
while True:
    success , img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h , w, c = img.shape
            cx , cy = int(lm.x*w) , int(lm.y*h)
            cv.circle(img,(cx,cy),10,(0,255,0),-1)
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),thickness=2)

    cv.imshow("Image",img)

    if cv.waitKey(1) & 0xff == ord("q"):
        break
cap.release()
cv.destroyAllWindows()


