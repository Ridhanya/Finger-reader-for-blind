import cv2

capture = cv2.CaptureFromCAM(0)
num = 0
while(true):
  img=cv.QueryFrame(capture)
  file=cv2.imwrite('pic'+str(num)+'.jpg',img)
  num+=1
  fr.fingertipdetection(file)
  fr.



