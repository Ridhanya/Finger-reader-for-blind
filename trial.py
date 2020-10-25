import cv2
import numpy as np
import tesserocr as tr
from PIL import Image
from math import sin,cos,radians
import imutils








mser = cv2.MSER_create()
img = cv2.imread('fr.jpg')
img = cv2.resize(img, (800,600), interpolation = cv2.INTER_AREA)        
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
vis = img.copy()
regions, _ = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img', vis)
cv2.imwrite("clear.jpg",vis)
cv2.waitKey(0)


cv_img = cv2.imread('clear.jpg', cv2.IMREAD_UNCHANGED)

  # since tesserocr accepts PIL images, converting opencv image to pil
pil_img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))

  #initialize api
api = tr.PyTessBaseAPI()
try:
  	# set pil image for ocr
  api.SetImage(pil_img)
    # Google tesseract-ocr has a page segmentation methos(psm) option for specifying ocr types
    # psm values can be: block of text, single text line, single word, single character etc.
    # api.GetComponentImages method exposes this functionality
    # function returns:
    # image (:class:`PIL.Image`): Image object.
    # bounding box (dict): dict with x, y, w, h keys.
    # block id (int): textline block id (if blockids is ``True``). ``None`` otherwise.
    # paragraph id (int): textline paragraph id within its block (if paraids is True).
    # ``None`` otherwise.
  boxes = api.GetComponentImages(tr.RIL.WORD,True)
    # get text
  text = api.GetUTF8Text()
    # iterate over returned list, draw rectangles
  for (im,box,_,_) in boxes:
      x,y,w,h = box['x'],box['y'],box['w'],box['h']
      cv2.rectangle(cv_img, (x,y), (x+w,y+h), color=(0,0,255))
finally:
  api.End()
height, width, channels = cv_img.shape
print height, width, channels


cv2.imshow('output', cv_img)
cv2.waitKey(0)
cv2.imwrite('mini1.jpg',cv_img)


image = cv2.imread('mini.jpg')
hsv= cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
mask = cv2.inRange(hsv, lower, upper)
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
c = max(cnts, key=cv2.contourArea)

extTop = tuple(c[c[:, :, 1].argmin()][0])

print extTop



v_img = cv2.imread('mini1.jpg', cv2.IMREAD_UNCHANGED)

  
pil_img = Image.fromarray(cv2.cvtColor(v_img,cv2.COLOR_BGR2RGB))
height, width, channels = v_img.shape
print height, width, channels 


cv2.line(v_img,(extTop[0],extTop[1]), (extTop[0],0), (255,0,0), 2)
cv2.imshow('line',v_img)
cv2.waitKey(0)


mg1 = np.zeros((512,512,3), np.uint8)
mg = cv2.resize(mg1, (800,600), interpolation = cv2.INTER_AREA)  
cv2.imshow('black',mg)
cv2.waitKey(0)

cv_img = cv2.imread('clear.jpg', cv2.IMREAD_UNCHANGED)

  # since tesserocr accepts PIL images, converting opencv image to pil
pil_img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))

  #initialize api
api = tr.PyTessBaseAPI()
try:
  	# set pil image for ocr
  api.SetImage(pil_img)
    # Google tesseract-ocr has a page segmentation methos(psm) option for specifying ocr types
    # psm values can be: block of text, single text line, single word, single character etc.
    # api.GetComponentImages method exposes this functionality
    # function returns:
    # image (:class:`PIL.Image`): Image object.
    # bounding box (dict): dict with x, y, w, h keys.
    # block id (int): textline block id (if blockids is ``True``). ``None`` otherwise.
    # paragraph id (int): textline paragraph id within its block (if paraids is True).
    # ``None`` otherwise.
  boxes = api.GetComponentImages(tr.RIL.WORD,True)
    # get text
  text = api.GetUTF8Text()
    # iterate over returned list, draw rectangles
  for (im,box,_,_) in boxes:
      x,y,w,h = box['x'],box['y'],box['w'],box['h']
      cv2.rectangle(mg, (x,y), (x+w,y+h), color=(0,0,255))
finally:
  api.End()

cv2.imshow('blines',mg)
cv2.waitKey(0)
cv2.imwrite('new.jpg',mg)

new = cv2.imread('new.jpg')
cv2.line(new,(extTop[0],extTop[1]), (extTop[0],0), (255,0,0), 2)

for (im,box,_,_) in boxes:
      x,y,w,h = box['x'],box['y'],box['w'],box['h']
      
      a=(y-h-128562)/(-287.5)
      b=y-h

      if(((b+(287.5*a)-128562)==0) and (((a-x+w)*(a-x)*(b-y+h)*(b-y+h))<=0)):
        cv2.rectangle(new, (x,y), (x+w,y+h), color=(255,255,255))
cv2.imshow('ty',new)
cv2.waitKey(0)        

imgray = cv2.cvtColor(new,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)




k=[]
for c in contours:
  for i in c:
  	for (j) in i:
  		if(j[0]==extTop[0]):
  			k.append(j[1])
  			l=max(k)

idx=0
for c in contours:
  for i in c:
  	for (j) in i:
  		if(j[0]==extTop[0] and j[1]==l):
  			x,y,w,h = cv2.boundingRect(c)
        new_img=image[y:y+h,x:x+w]
        cv2.imwrite('let.jpg', new_img)
        new_img=image[y:y+h,x:x+w]
        cv2.imwrite('let.jpg', new_img)
        
        
        
  		
      
cv2.imshow('np',new_img)
cv2.waitKey(0)
cv2.imshow('OUTPUT',image)
cv2.waitKey(0)
cv2.imshow('mask',mask)
cv2.waitKey(0)        
        
cv2.imshow('new',new)
cv2.waitKey(0)  			   






  			

            

      
cv2.imshow('con',new)
cv2.waitKey(0)
cv2.imshow('suc',image)
cv2.waitKey(0)

cv2.imshow('c',new)
cv2.waitKey(0)
cv2.imwrite('conc.jpg',new)
cv2.imwrite('suc.jpg',image)






cv2.destroyAllWindows()
