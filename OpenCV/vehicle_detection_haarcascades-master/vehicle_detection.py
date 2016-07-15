# -*- coding: utf-8 -*-

#Only detecs cars.

import cv2
import numpy as np
# print cv2.__file__
print(cv2.__version__)

cascade_src = 'cars.xml'
#video_src = '/home/msit/Presentation/OpenCV/vehicle_detection_haarcascades-master/dataset/video2.avi'
#video_src = '/home/msit/dataCollection/slavePi2_RW1600_RH1200_TT60_FR15_06_03_2016_17_01_45_618527.h264'

cap = cv2.VideoCapture("slavePi4_RW1600_RH1200_TT180_FR15_06_03_2016_13_53_54_254155.h264")
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, frame = cap.read()
    if (type(frame) == type(None)):
        print "is true"
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255

    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minSize=(90, 90), maxSize=(1000, 1000))

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('video', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#cv2.destroyAllWindows()