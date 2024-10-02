import cv2 as cv
import mediapipe as mp
import time


cap = cv.VideoCapture(0)
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils
pTime = 0
while True:
    success , img = cap.read()
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id , detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih , iw, ic = img.shape
            bbox = int(bboxC.xmin * iw) , int(bboxC.ymin * ih) , \
                    int(bboxC.width * iw) , int(bboxC.height * ih)
            cv.rectangle(img,bbox,(0,255,0),thickness=2)
            cv.putText(img,f'{str(int(detection.score[0]*100))}%',(bbox[0],bbox[1]-20),
                       cv.FONT_HERSHEY_COMPLEX,
                       0.5 , (0,255,0))


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),thickness=2)
    cv.imshow("Image",img)
    if cv.waitKey(20) & 0xff == ord("q"):
        break

cap.release()
cv.destroyAllWindows()