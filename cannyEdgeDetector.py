## Exploration of the edge detection function Canny() which uses the Canny algorithm to determine edges within an image.
# This is a cv2.Canny() example from:
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
# Added this project to GitHub

import cv2
import numpy as np
from matplotlib import pyplot as plt

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
