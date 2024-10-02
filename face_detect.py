import cv2 as cv

img = cv.imread("D:\\Python Projects\\pythonProject\\Group Photo (3).jpg")
img = cv.resize(img,(700,800))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

hc = cv.CascadeClassifier("haarcascade.xml")
fr = hc.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3)
print(len(fr))

for (x,y,w,h) in fr:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv.imshow("Detected Image",img)
cv.waitKey(0)
cv.destroyAllWindows()