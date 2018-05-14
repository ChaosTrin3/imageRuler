## This file attempts to locate a rectangular object and determine the position and orientation

import cv2
import numpy as np

img = cv2.imread("mock-foceplate_90-01.jpg", 0)  #load an image, the 0 denotes importing as grayscale
#img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to grayscale, another way to convert to grayscale
ret, thresh = cv2.threshold(img, 250, 1, cv2.THRESH_BINARY) #binary threshold, !currently getting held up here...
thresh = 1 - thresh  #invert: 1 for the battery, 0 for the background

h, w = thresh.shape

#From a matrix of pixels to a matrix of coordinates of non-black points.
#(note: mind the col/row order, pixels are accessed as [row, col]
#but when we draw, it's (x, y), so have to swap here or there)

mat = [[col, row] for col in range(w) for row in range(h) if thresh[row, col] != 0]
mat = np.array(mat).astype(np.float32)#have to convert type for PCA

#mean (e. g. the geometrical center)
#and eigenvectors (e. g. directions of principal components)
m, e = cv2.PCACompute(mat, mean = np.array([]))

#now to draw: let's scale our primary axis by 100,
#and the secondary by 50
center = tuple(m[0])
endpoint1 = tuple(m[0] + e[0]*100)
endpoint2 = tuple(m[0] + e[1]*50)

red_color = (0, 0, 255)
cv2.circle(img, center, 5, red_color)
cv2.line(img, center, endpoint1, red_color)
cv2.line(img, center, endpoint2, red_color)
cv2.imwrite("out.png", img)
