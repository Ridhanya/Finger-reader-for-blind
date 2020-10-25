import cv2
import numpy as np

mser = cv2.MSER_create()
img = cv2.imread('fr.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()
regions, _ = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('img', vis)
cv2.waitKey(0)
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

text_only = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow("text only", text_only)
cv2.waitKey(0)


