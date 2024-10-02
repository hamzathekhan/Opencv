import numpy as np
import cv2 as cv

people = ['Dua Lipa', 'Elon Musk', 'Ronaldo', 'Salman Khan']
haar_cascade = cv.CascadeClassifier("haarcascade.xml")

# features = np.load("features.npy")
# labels = np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")


img = cv.imread(r'Validation/download (3).jfif')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Person",gray)

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

for (x,y,w,h) in faces_rect:
    face_roi = gray[y:y+h,x:x+w]

    label , confidence = face_recognizer.predict(face_roi)
    print(f'Label = {label} with a Confidence of {confidence}')

    cv.putText(img,str(people[label]),(10,130),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow("Detected Face",img)

cv.waitKey(0)
cv.destroyAllWindows()