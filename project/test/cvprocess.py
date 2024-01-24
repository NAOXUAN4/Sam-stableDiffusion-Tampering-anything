import cv2
import numpy as np


def cv2show(img):
    img = cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
    cv2.imshow("img",img)
    cv2.waitKey()



img = cv2.imread('result.jpg')

kernel=np.ones((10,10),dtype=np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


cv2show(img)

