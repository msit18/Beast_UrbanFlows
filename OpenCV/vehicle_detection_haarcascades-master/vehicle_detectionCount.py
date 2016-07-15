# -*- coding: utf-8 -*-

#Attempted to integrate car detection and optical flows

import cv2
import numpy as np
import video
from common import anorm2, draw_str
from time import clock
import time

cascade_src = 'cars.xml'
#video_src = '/home/msit/Presentation/OpenCV/vehicle_detection_haarcascades-master/dataset/video2.avi'
#video_src = '/home/msit/dataCollection/slavePi2_RW1600_RH1200_TT60_FR15_06_03_2016_17_01_45_618527.h264'

#cap = cv2.VideoCapture("slavePi4_RW1600_RH1200_TT180_FR15_06_03_2016_13_53_54_254155.h264")
cap = cv2.VideoCapture("/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_10_2016_18_11_00_604698.avi")
car_cascade = cv2.CascadeClassifier(cascade_src)

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 10
        self.tracks = []
        self.frame_idx = 0
        self.rectangleTracks = []
        self.totalNumCars = 0

    def run(self):
        try:
            while True:
                ret, frame = cap.read()
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()

                #cv2.line(vis, (900, 300), (200, 600), (0, 0, 255), 3)
                #cv2.rectangle(vis, (200, 600), (1100, 300), (0, 0, 255), 3)

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag, (rectX, rectY, rectW, rectH) in zip(self.tracks, p1.reshape(-1, 2), good, self.rectangleTracks):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                        cv2.rectangle(vis,(rectX,rectY),(rectX+rectW,rectY+rectH),(0,0,255),2)

                        #Is it better to do a ROI or should I do a line in front of all the traffic things?
                        #In any case, either do within area or if the coordinates pass the line
                        #THIS DOESN'T WORK BY THE WAY. NEED TO FIGURE OUT HOW TO DETERMINE IF THE SAME POINT HAS ALREADY
                        #ENTERED THE BOX. OR SHOULD I JUST DO A LINE? HOW TO VERIFY.
                        # if ((int(x) > 200 & int(x) < 1100) & (int(y) > 300 & int(y) < 600)):
                        #     self.totalNumCars += 1
                        #     print "total number of cars detected: ", self.totalNumCars

                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
                    #draw_str(vis, (40, 40), 'car count: %d' % (self.totalNumCars))

                # if self.frame_idx % self.detect_interval == 0:
                self.rectangleTracks = []
                # mask = np.zeros_like(frame_gray)
                # mask[:] = 255
                # for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                #     cv2.circle(mask, (x, y), 5, 0, -1)
                # p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                cars = car_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minSize=(100, 100), maxSize=(800, 800))
                #print len(cars)
                for (x,y,w,h) in cars:
                    car_centroid = [(x+(w/2)), (y+(h/2))]
                    #cv2.circle (vis, ( (x+(w/2)) , (y+(h/2)) ), 4, (255, 0, 0), 4)
                    self.rectangleTracks.append([x,y,w,h])
                    # self.tracks = np.append(self.tracks, car_centroid, axis=0)
                    # p0 = np.float32(self.tracks).reshape(-1, 1, 2)
                #print "p: \n", car_centroid
                    if car_centroid is not None:
                        # for x, y in np.float32(car_centroid).reshape(-1, 2):
                        #     self.tracks.append([(x, y)])
                        for a, b in np.float32(car_centroid).reshape(-1, 2):
                            self.tracks.append([(a, b)])
                                

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)

                ch = 0xFF & cv2.waitKey(1)
                if ch == 27:
                    break
        finally:
            end = time.time()
            print "end: ", end
            return end

if __name__ == '__main__':
    start = time.time()
    print "start: ", start
    end = App(cap).run()
    #end = time.time()
    print "end: ", end
    totalTime = end-start
    print "totaltime to process was: ", totalTime
    cv2.destroyAllWindows()
