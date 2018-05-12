## Exploration of the edge detection function Canny() which uses the Canny algorithm to determine edges within an image.
# This is a cv2.Canny() example from:
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
# Added this project to GitHub

import cv2
import numpy as np
from matplotlib import pyplot as plt

#pathToHaarCascades = 'C:/Users/jdt01/Downloads/opencv/sources/data/haarcascades/'
# The path to the required datasets are added to the repo specifically..
pathToHaarCascades = ''

#while True:
lowLim = int(input("lowLim of the Canny hysteresis input: "))
highLim = int(input("highLim of the Canny hysteresis input: "))

img = cv2.imread('tooling-plate.jpg',0)
edges = cv2.Canny(img,lowLim,highLim)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

# Object recognition, although it is using a standard face database...
# The core tech will be used to find pockets relative to edges..

face_cascade = cv2.CascadeClassifier(pathToHaarCascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(pathToHaarCascades + 'haarcascade_eye.xml')

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
