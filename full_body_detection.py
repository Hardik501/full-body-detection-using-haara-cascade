import numpy as np
import cv2
from matplotlib import pyplot as plt
bodydetection = cv2.CascadeClassifier('haarcascade_fullbody.xml')
ubodydetection = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
lbodydetection = cv2.CascadeClassifier('haarcascade_upperbody.xml')
#img = cv2.imread('p1.jpg')
#img = cv2.imread('p2.jpg')
img = cv2.imread('p3.jpg')
## for live camera detection replace 'img' with frame in belowed code
#cam=cv2.VideoCapture(0)
#while True:
#    ret, frame = cam.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
body=bodydetection.detectMultiScale(gray,1.1,1)
#print lips
for (a,b,c,d) in body:
    cv2.rectangle(img,(a,b),(a+c,b+d),(0,0,255),2)
upper=ubodydetection.detectMultiScale(gray,1.1,1)
for (a,b,c,d) in upper:
    cv2.rectangle(img,(a,b),(a+c,b+d),(0,0,255),2)
lower=lbodydetection.detectMultiScale(gray,1.1,1)
for (a,b,c,d) in lower:
    cv2.rectangle(img,(a,b),(a+c,b+d),(0,0,255),2)
cv2.imshow('detect',img)
cv2.waitKey(0)
##for live camera detection
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
#cv2.destroyAllWindows()