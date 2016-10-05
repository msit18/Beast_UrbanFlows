#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Original Example Code by Ross Girshick

# This code is work created by Michelle Sit
# --------------------------------------------------------

"""
Demo script showing detections in sample images.
INTEGRATING TRACKING FOR THESE CARS

"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, time, math
import argparse

import video
from common import anorm2, draw_str
from time import clock

class UrbanFlows():

    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 500,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

#Detects cars within frame, detects corners of interest within detected car frame, outputs array
    def carDetectionMethod(self, im, im_copy, frame_gray, net, detectedCarsInThisFrame):
        scores, boxes = im_detect(net, im)

        # Visualize detections for each class
        CONF_THRESH = 0.7
        NMS_THRESH = 0.3

        detectedCarsInThisFrame = []

        for cls_ind, cls in enumerate(CLASSES[1:]):
            #detect all potential elements of interest here using rcnn
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32) #stacks them together
            keep = nms(dets, NMS_THRESH) #Removes overlapping bounding boxes
            dets = dets[keep, :]
            if cls == "car":
                inds = np.where(dets[:, -1] >= 0.5)[0] #Threshold applied to score values here
                # if len(inds) == 0:
                #     return tracks

                im = im[:, :, (2, 1, 0)]

                #Calculate center of box, and draw on image. Use cvDrawBBox for cv2 (expects integers)
                #x = bbox[0], y = bbox[1] (top left corner)
                #x1 = bbox[2], y1 = bbox[3] (bottom right corner)
                #print "Number of cars in frame: ", len(inds)
                for i in inds:
                    bbox = dets[i, :4]
                    carDetectedBBox = bbox.astype(int)
                    score = dets[i, -1]
                    #Draw rectangle on color copy
                    cv2.rectangle(im_copy, (carDetectedBBox[0], carDetectedBBox[1]), (carDetectedBBox[2], carDetectedBBox[3]), (255, 0, 0), 3)
                    
                    #Calculate corners of interest within the bounding box area and add them all to the carCorner array
                    detectedCarPixels = frame_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]] #[y1:y2, x1:x2]
                    detectedCarPixelsColor = im_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]] #for show on colored image
                    carCorners = cv2.goodFeaturesToTrack(detectedCarPixels, mask=detectedCarPixels, **self.feature_params).reshape(-1, 2)

                    for x, y in np.float32(carCorners).reshape(-1, 2): #Blue
                        cv2.circle(detectedCarPixels, (x,y), 5, (255, 0, 0), -1)
                        cv2.circle(detectedCarPixelsColor, (x, y), 5, (255, 0, 0), -1)

                    detectedCarsInThisFrame.append([[carDetectedBBox], [carCorners]])

                print "detectedCarsInThisFrame len: {0}-------------------------------------".format(len(detectedCarsInThisFrame))

                return detectedCarsInThisFrame

#     def tempThresholding2 (self, detectedCars, newCars):
#         newCarArray = []
#         for singleNewCar in range(len(newCars)):
#             for singleDetectedCar in range(len(detectedCars)):
#                 print "math: ", math.hypot(abs(detectedCars[singleDetectedCar][0][0][0] - newCars[singleNewCar][0][0][0]), abs(detectedCars[singleDetectedCar][0][0][1] - newCars[singleNewCar][0][0][1]))
#                 if math.hypot(abs(detectedCars[singleDetectedCar][0][0][0] - newCars[singleNewCar][0][0][0]), abs(detectedCars[singleDetectedCar][0][0][1] - newCars[singleNewCar][0][0][1])) < 100:
#                     detectedCars[singleDetectedCar][0].append(newCars[singleNewCar][0][0])
#                     detectedCars[singleDetectedCar][1].append(newCars[singleNewCar][1])
#                     newCarArray.append(detectedCars[singleDetectedCar])
#                     del detectedCars[singleDetectedCar]
#                     print "newCarArray shape: ", np.array(newCarArray).shape
#                     print "detectedCars shape: ", np.array(detectedCars).shape
#                     break
#             print "singleDC00: ", detectedCars[singleDetectedCar][0][0]
#             print "detectedCar[-1][0]: ", detectedCars[-1][0]
#             print "dC-10: ", detectedCars[-1][0][0]
#             print "print if statement: ", all((detectedCars[singleDetectedCar][0][0] == detectedCars[-1][0][0]))
#             if all(detectedCars[singleDetectedCar][0][0]== detectedCars[-1][0][0]):
#                 print "appending to the end"
#                 newCarArray.append(newCars[singleNewCar])
#             print "nextCar:::::::::::::::::::::::;"

#         print "len newCar now: ", len(newCars)
#         print "len detectedCars now: ", len(detectedCars)
#         print "len newCarArray now: ", len(newCarArray)
#         print "REMAINING CARS-------------"
#         if len(detectedCars) > 0:
#             for remainingCar in detectedCars:
#                 #print "shape of remainingCar: ", np.array(remainingCar).shape
#                 #print "shape of newCarArray before append: ", np.array(newCarArray).shape
#                 newCarArray.append(remainingCar)
#                 #print "shape of newCarArray after append: ", np.array(newCarArray).shape

#         print "-----------------------shape of newCar array: ", np.array(newCarArray).shape
#         for x in range(len(newCarArray)):
#             print "x: ", newCarArray[x][0]

#         return newCarArray


# #Compare the cars in newCars with the ones in detectedCars to determine if they are the same car. Append to correct places.
# #There are duplicates in detectedCars for some reason. Need to catch why. Need to carefully examine detectedCars and newCarArray
#     def tempThresholding3 (self, detectedCars, newCars):
#         newCarArray = []
#         for singleNewCar in newCars:
#             for singleDetectedCar in detectedCars:
#                 print "math: ", math.hypot(abs(singleDetectedCar[0][0][0] - singleNewCar[0][0][0]), abs(singleDetectedCar[0][0][1] - singleNewCar[0][0][1]))
#                 if math.hypot(abs(singleDetectedCar[0][0][0] - singleNewCar[0][0][0]), abs(singleDetectedCar[0][0][1] - singleNewCar[0][0][1])) < 100:
#                     singleDetectedCar[0].append(singleNewCar[0][0])
#                     singleDetectedCar[1].append(singleNewCar[1])
#                     newCarArray.append(singleDetectedCar)
#                     all(detectedCars.remove(singleDetectedCar))

#                     # # print "singleDetectedCar to remove from DetectedCar Array0: ", singleDetectedCar[0]
#                     # # print "detectedCars shape before: ", np.array(detectedCars).shape
#                     # for car in range(len(detectedCars)):
#                     #     # print "Car car: ", detectedCars[car]
#                     #     # print "singleDetectedCar: ", singleDetectedCar
#                     #     # print "if statement: ", detectedCars[car] == singleDetectedCar
#                     #     if detectedCars[car] == singleDetectedCar:
#                     #         del detectedCars[car]
#                     #         break
#                     #del detectedCars[x] for x in range(len(detectedCars)) if car == singleDetectedCars
#                     print "newCarArray shape: ", np.array(newCarArray).shape
#                     print "detectedCars shape: ", np.array(detectedCars).shape
#                     break
#             print "singleDC00: ", singleDetectedCar[0][0]
#             print "dC-10: ", detectedCars[-1][0]
#             print "detectedCar[-1]: ", detectedCars[-1]
#             print "print if statement: ", all(singleDetectedCar[0] == detectedCars[-1][0])
#             if all(singleDetectedCar[0] == detectedCars[-1][0]):
#                 print "appending to the end"
#                 newCarArray.append(singleNewCar)
#             print "nextCar:::::::::::::::::::::::;"

#         print "len newCar now: ", len(newCars)
#         print "len detectedCars now: ", len(detectedCars)
#         print "len newCarArray now: ", len(newCarArray)
#         print "REMAINING CARS-------------"
#         if len(detectedCars) > 0:
#             for remainingCar in detectedCars:
#                 #print "shape of remainingCar: ", np.array(remainingCar).shape
#                 #print "shape of newCarArray before append: ", np.array(newCarArray).shape
#                 newCarArray.append(remainingCar)
#                 #print "shape of newCarArray after append: ", np.array(newCarArray).shape

#         print "-----------------------shape of newCar array: ", np.array(newCarArray).shape
#         for x in range(len(newCarArray)):
#             print "x: ", newCarArray[x][0]

#         return newCarArray

#need to think this over some more. Last write 9/27
#Need to optimize this more later: for now check entire frame and all the points. Later, crop frame and only certain points.
#Later only track for x number of frames and update
#Given corner points of interest, find corners to track in entire frame and past frame
#If time later, reshape singleCarPoints to have one less array surrounding it
    # #def trackCars(self, detectedCars, frame_gray, prev_gray, im_copy):
    #     for singleCarPoints in detectedCars:
    #         print "SingleCarPoints first values: ", np.float32(singleCarPoints[0][:])
    #         print "SingleCarPoints second values: ", np.float32(singleCarPoints[1][:])

    #         drawBoxCoord = np.float32(singleCarPoints[0][:]).astype(int)
    #         drawBoxFrame = im_copy[drawBoxCoord[1]:drawBoxCoord[3], drawBoxCoord[0]:drawBoxCoord[2]]
    #         drawBoxGrayFrame = frame_gray[drawBoxCoord[1]:drawBoxCoord[3], drawBoxCoord[0]:drawBoxCoord[2]]

    #         p0 = np.float32(singleCarPoints[1][:])
    #         p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **self.lk_params)
    #         p0r, st, err = cv2.calcOpticalFlowPyrLK(frame_gray, prev_gray, p1, None, **self.lk_params)

    #         #Visualizations to understand what the points mean. Delete later
    #         # for val in range(len(p0)): #Green
    #         #     cv2.circle(drawBoxGrayFrame, (p0[val][0][0], p0[val][0][1]), 5, (0, 255, 0), -1)
    #         #     cv2.circle(drawBoxFrame, (p0[val][0][0], p0[val][0][1]), 5, (0, 255, 0), -1)

    #         # cv2.imshow('DrawBox', drawBoxFrame)
    #         # if cv2.waitKey(0) & 0xFF == ord('q'):
    #         #     break

    #         # for val in range(len(p1)): #Red
    #         #     cv2.circle(drawBoxGrayFrame, (p1[val][0][0], p1[val][0][1]), 5, (0, 0, 255), -1)
    #         #     cv2.circle(drawBoxFrame, (p1[val][0][0], p1[val][0][1]), 5, (0, 0, 255), -1)

    #         for val in range(len(p0r)): #Pink
    #             cv2.circle(drawBoxGrayFrame, (p0r[val][0][0], p0r[val][0][1]), 5, (255, 0, 255), -1)
    #             cv2.circle(drawBoxFrame, (p0r[val][0][0], p0r[val][0][1]), 5, (255, 0, 255), -1)

    #         #cv2.imshow('DrawBox', drawBoxFrame)
    #         cv2.imshow('p values', im_copy)
    #         if cv2.waitKey(0) & 0xFF == ord('q'):
    #             break

    #         #This is the part of the algorithm that determines if the points are close enough to be the same object
    #         d = abs(p0-p0r).reshape(-1, 2).max(-1)
    #         print "D: ", d
    #         print "shape D: ", d.shape
    #         print "len D: ", len(d)
    #         good = d < 1
    #         print "good: ", good
    #         print "p1.reshape all: ", p1.reshape(-1, 2)
    #         # for tr, (x, y), good_flag in zip(singleCarPoints, p1.reshape(-1, 2), good):
    #         for (x, y), good_flag in zip(p1.reshape(-1, 2), good):
    #             if not good_flag:
    #                 print "p1.reshape not good: {0}, {1}".format(x, y)
    #                 print "good_flag: ", good_flag
    #                 continue
    #             print "P1.reshape : {0}, {1}".format(x, y)
    #             print "good_flag: ", good_flag
    #             # tr.append((x, y))
    #             # if len(tr) > self.track_len:
    #             #     del tr[0]
    #         print "end of track cars-----------------------------------------------------------"

    #Given points of different cars, check how many overlapping points there are and if you can detect different cars from the overlap
    #Instead of providing frame_gray and prev_gray, given different images, calc the p1 vals
    #Error: two images aren't the same size. How to fix this? 9/27**
    #def checkSimilarCars(self, detectedCars, frame_gray, prev_gray, im_copy):
    #     print "detectedCars0: ", detectedCars[0]
    #     drawBoxCoord0 = np.float32(detectedCars[0][0][:]).astype(int)
    #     print "drawBoxCoord0: ", drawBoxCoord0
    #     drawBoxFrame0 = im_copy[drawBoxCoord0[1]:drawBoxCoord0[3], drawBoxCoord0[0]:drawBoxCoord0[2]]
    #     drawBoxGrayFrame0 = frame_gray[drawBoxCoord0[1]:drawBoxCoord0[3], drawBoxCoord0[0]:drawBoxCoord0[2]]

    #     #cv2.imshow('drawBoxFrame0', drawBoxFrame0)

    #     print "detectedCars1: ", detectedCars[1]
    #     drawBoxCoord1 = np.float32(detectedCars[1][0][:]).astype(int)
    #     print "drawBoxCoord1: ", drawBoxCoord1
    #     drawBoxFrame1 = im_copy[drawBoxCoord1[1]:drawBoxCoord1[3], drawBoxCoord1[0]:drawBoxCoord1[2]]
    #     drawBoxGrayFrame1 = frame_gray[drawBoxCoord1[1]:drawBoxCoord1[3], drawBoxCoord1[0]:drawBoxCoord1[2]]

    #     #cv2.imshow('drawBoxFrame1', drawBoxFrame1)

    #     p0 = np.float32(detectedCars[0][1][:])
    #     print "p0: ", p0
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(drawBoxGrayFrame1, drawBoxGrayFrame0, p0, None, **self.lk_params)
    #     print "p1: ", p1
    #     p0r, st, err = cv2.calcOpticalFlowPyrLK(drawBoxGrayFrame0, drawBoxGrayFrame1, p1, None, **self.lk_params)


    # #def exampleCode(self, detectedCars, frame_gray, prev_gray): 
    #     img0, img1 = self.prev_gray, frame_gray
    #     p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    #     p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    #     d = abs(p0-p0r).reshape(-1, 2).max(-1)
    #     good = d < 1
    #     new_tracks = []
    #     for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
    #         if not good_flag:
    #             continue
    #         tr.append((x, y))
    #         if len(tr) > self.track_len:
    #             del tr[0]
    #         new_tracks.append(tr)
    #         cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
    #     self.tracks = new_tracks
    #     cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
    #     draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

    def thresholding (self, detectedCars, newCars):
        aggregatedCars = []
        print "detectedCars shape: ", np.array(detectedCars).shape
        print "newCars shape: ", np.array(newCars).shape
        for singleNewCarIndex in range(len(newCars)):
            for singleDetectedCarIndex in range(len(detectedCars)):
                confirmAppended = False
                print "math: ", math.hypot(abs(newCars[singleNewCarIndex][0][0][0] - detectedCars[singleDetectedCarIndex][0][0][0]), abs(newCars[singleNewCarIndex][0][0][1] - detectedCars[singleDetectedCarIndex][0][0][1]))
                if math.hypot(abs(newCars[singleNewCarIndex][0][0][0] - detectedCars[singleDetectedCarIndex][0][0][0]), abs(newCars[singleNewCarIndex][0][0][1] - detectedCars[singleDetectedCarIndex][0][0][1])) < 100:
                    print "Less than 100"
                    detectedCars[singleDetectedCarIndex].append(newCars[singleNewCarIndex])
                    aggregatedCars.append(detectedCars[singleDetectedCarIndex])
                    del detectedCars[singleDetectedCarIndex]
                    confirmAppended = True
                    print "detectedCars shape after: ", np.array(detectedCars).shape
                    print "aggregatedCars shape after: ", np.array(aggregatedCars).shape
                    print "newCars shape after: ", np.array(newCars).shape
                    break
            if (confirmAppended == False) & (singleNewCarIndex < len(newCars)-1):
                aggregatedCars.append(newCars[singleNewCarIndex])
                print "If statement is true. Appended size is now: ", np.array(aggregatedCars).shape
            print "NEXT CAR;;;;;;;;;;;;;;;;;;;;;;;;;;;;"

        print "len newCars: ", len(newCars)
        print "len detectedCars: ", len(detectedCars)
        print "len aggregatedCars: ", len(aggregatedCars)
        print "remaining cars-------------------"
        if len(detectedCars)>0:
            for remainingCar in detectedCars:
                aggregatedCars.append(remainingCar)

        print "-----------------------shape of newCar array: ", np.array(aggregatedCars).shape
        for x in range(len(aggregatedCars)):
            print "x: ", aggregatedCars[x][0]

        return aggregatedCars

#Simple code - works crudely. No temporal tracking. Need to use corners to track
    def simpleThresholding(self, tracks, inputArray):
        tracksLength = len(tracks)
        for x in range(len(inputArray)):
            print str(inputArray[x][-1][0]) + "-------------------------------"
            newTracksXVal = inputArray[x][-1][0]
            newTracksYVal = inputArray[x][-1][1]
            confirmAppended = False
            for y in range(tracksLength):
                print "math: ", math.hypot(newTracksXVal - tracks[y][-1][0], newTracksYVal - tracks[y][-1][1])
                if math.hypot(newTracksXVal-tracks[y][-1][0], newTracksYVal-tracks[y][-1][1]) <= 100:
                    tracks[y].append((newTracksXVal, newTracksYVal))
                    print "tracks appended in that value: ", tracks
                    break
                if len(tracks[y]) > 10:
                    print "track too long. deleting {0}".format(tracks[y][0])
                    del tracks[y][0]
                elif (confirmAppended == False) & (y == tracksLength-1):
                    tracks.append([(newTracksXVal, newTracksYVal)])
                    print "tracks appended to the end"

    def detectTrackCars(self, net):
        """Detect object classes in an image using pre-computed object proposals."""

        cap = cv2.VideoCapture("/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_10_2016_18_11_00_604698.h264")

        track_len = 10
        #corners = [] #stores all coordinates of corners of interest for detected cars that pass the threshold
        #rectangleTracks = [] #stores the bounding box coordinates for each detected car
        detectedCars = [] #stores all information about the detected cars
        newCars = [] #newly detectedCars. Matched to ones stored in detectedCars in thresholding method
        
        while (cap.isOpened()):
            ret, im = cap.read()
            frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_copy = im.copy()
            #TO DO: See if I can remove an if statement, given the placement of prev_gray
            if len(detectedCars) <= 0:
                detectedCars = c.carDetectionMethod(im, im_copy, frame_gray, net, detectedCars)
            elif len(detectedCars) > 0:
                #print "detectTrackCars Method len newCars: ", len(newCars)
                print "detectedcars len: ", len(detectedCars)
                newCars = c.carDetectionMethod(im, im_copy, frame_gray, net, detectedCars)
                print "Updated newCars value: ", len(newCars)
                detectedCars = c.thresholding(detectedCars, newCars)

            print "before prev_gray"
            prev_gray = frame_gray

            # cv2.imshow('frame', im_copy)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

            print "take prev_gray frame****************************************************"

        cap.release()

    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Faster R-CNN demo')
        parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                            default=0, type=int)
        parser.add_argument('--cpu', dest='cpu_mode',
                            help='Use CPU mode (overrides --gpu)',
                            action='store_true')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                            choices=NETS.keys(), default='vgg16')

        args = parser.parse_args()

        return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    c = UrbanFlows()

    CLASSES = ('__background__',
                'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    NETS = {'vgg16': ('VGG16',
                      'VGG16_faster_rcnn_final.caffemodel'),
            'zf': ('ZF',
                      'ZF_faster_rcnn_final.caffemodel')}

    args = c.parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    c.detectTrackCars (net)
    cv2.destroyAllWindows()
