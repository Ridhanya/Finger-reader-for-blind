import cv2
import numpy as np
import tesserocr as tr
from PIL import Image
from math import sin,cos,radians

cv_img = cv2.imread('fr.jpg', cv2.IMREAD_UNCHANGED)

  
pil_img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))

  
api = tr.PyTessBaseAPI()
try:
  	
  api.SetImage(pil_img)
   
  boxes = api.GetComponentImages(tr.RIL.WORD,True)
  #boxes = api.GetComponentImages(tr.RIL.TEXTLINE,True)
      
  text = api.GetUTF8Text()
    
  for (im,box,_,_) in boxes:
      x,y,w,h = box['x'],box['y'],box['w'],box['h']
      cv2.rectangle(cv_img, (x,y), (x+w,y+h), color=(0,0,255))
finally:
  api.End()
height, width, channels = cv_img.shape
print height, width, channels
angle = -5150;
length = height;
x1=446
y1=337


x2 = int(round(x1 + length * cos(radians(angle * 3.14 / 180.0))))
y2 = int(round(y1 + length * sin(radians(angle * 3.14 / 180.0))))
cv2.line(cv_img,(x1,y1), (x2,y2), (0,255,0), 2)


cv2.imshow('output', cv_img)
cv2.waitKey(0)
cv2.imwrite('mini1.jpg',cv_img)
img = cv2.imread('mini1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines3.jpg',img)

cv2.destroyAllWindows()