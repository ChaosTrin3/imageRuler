import numpy as np
import cv2

# Now let's try and do some object recognition..
# OpenCV allows you to train your own datasets to detect any object you'd like
# but just for the sake of learning the tool quickly, let's use a pretrianed
# data set, detecting faces.

#It is important to note that the path here is after downloading the openCV .exe and extacting in a lazy location...
face_cascade = cv2.CascadeClassifier('C:/Users/jdt01/Downloads/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/jdt01/Downloads/opencv/sources/data/haarcascades/haarcascade_eye.xml')

img = cv2.imread('face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
