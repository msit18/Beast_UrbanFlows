# -*- coding: utf-8 -*-

#OpticalFlow and Vehicle detection are not integrated

import cv2
import numpy as np
import video
from common import anorm2, draw_str
from time import clock

cascade_src = 'cars.xml'
#video_src = '/home/msit/Presentation/OpenCV/vehicle_detection_haarcascades-master/dataset/video2.avi'
#video_src = '/home/msit/dataCollection/slavePi2_RW1600_RH1200_TT60_FR15_06_03_2016_17_01_45_618527.h264'

cap = cv2.VideoCapture("slavePi4_RW1600_RH1200_TT180_FR15_06_03_2016_13_53_54_254155.h264")
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
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.centroids = [1,1]

    def run(self):
        while True:
            ret, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            cars = car_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minSize=(90, 90), maxSize=(800, 800))
            print len(cars)

            for (x,y,w,h) in cars:
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                #print p0
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                # cars = car_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minSize=(90, 90), maxSize=(1000, 1000))
                # for (x,y,w,h) in cars:
                #     cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)
                #     car_centroid = [(x+(x/2)), (y+(y/2))]
                #     # self.tracks = np.append(self.tracks, car_centroid, axis=0)
                #     # p0 = np.float32(self.tracks).reshape(-1, 1, 2)
                #print "p: \n", car_centroid
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

if __name__ == '__main__':
    App(cap).run()
    cv2.destroyAllWindows()
