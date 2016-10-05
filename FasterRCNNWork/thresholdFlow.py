#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Original Example Code by Ross Girshick

# This code is work created by Michelle Sit
# THIS VERSION DOES NOT USE THE OPENCV OPTICAL FLOW ALGORITHM
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

#TODO: REWRITE USING NUMPY

class UrbanFlows():

    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict( maxCorners = 500,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    def detectCars(self, im, im_copy, frame_gray, net, tracks, numFrames):
        scores, boxes = im_detect(net, im)

        # Visualize detections for each class
        CONF_THRESH = 0.7
        NMS_THRESH = 0.3

        newDetectedCarPoints = []

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
                #x1 = bbox[2], y1 = bbox[3] (top right corner)
                # print "Number of cars in frame: ", len(inds)
                for i in inds:
                    bbox = dets[i, :4]
                    cvDrawBBox = bbox.astype(int)
                    score = dets[i, -1]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    cv2.rectangle(im_copy, (cvDrawBBox[0], cvDrawBBox[1]), (cvDrawBBox[2], cvDrawBBox[3]), (255, 0, 0), 3)
                    
                    car_centroid = [(bbox[0]+(w/2)), (bbox[1]+(h/2))]
                    cv2.circle (im_copy, ( (cvDrawBBox[0]+(w.astype(int)/2)) , (cvDrawBBox[1]+(h.astype(int)/2)) ), 4, (255, 0, 0), 4)
                    if car_centroid is not None:
                        for a, b in np.float32(car_centroid).reshape(-1, 2):
                            print "car coordinates: ({0}, {1})".format(a, b)
                            newDetectedCarPoints.append([(a, b)])

                print "Tracks: ", tracks
                print "New Tracks: ", newDetectedCarPoints

                if len(tracks) > 0:
                    c.thresholding(tracks, newDetectedCarPoints)
                    # if numFrames%10 ==0:
                    #     c.temporalRemove(tracks)
                    #     numFrames = 0
                else:
                    # print "running else"
                    tracks = newDetectedCarPoints

                numFrames += 1
                # print "numFrames: ", numFrames
                newDetectedCarPoints = []
                # cv2.imshow('rectangles', im_copy)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                return tracks

#Need to figure out a temporal method of removing old points
    def thresholding(self, tracks, inputArray):
        tracksLength = len(tracks)
        for x in range(len(inputArray)):
            print str(inputArray[x][-1][0]) + "-------------------------------"
            newTracksXVal = inputArray[x][-1][0]
            newTracksYVal = inputArray[x][-1][1]
            confirmAppended = False
            print "range(tracksLength): ", tracksLength
            for y in range(tracksLength):
                print "track[{0}] len: ".format(y), len(tracks[y])
                # print "math: ", math.hypot(newTracksXVal - tracks[y][-1][0], newTracksYVal - tracks[y][-1][1])
                if math.hypot(newTracksXVal-tracks[y][-1][0], newTracksYVal-tracks[y][-1][1]) <= 100:
                    tracks[y].append((newTracksXVal, newTracksYVal))
                    print "tracks appended in that value: ", tracks
                    break
                elif len(tracks[y]) > 10:
                    print "track too long. deleting {0}".format(tracks[y][0])
                    del tracks[y][0]
                elif (confirmAppended == False) & (y == tracksLength-1):
                    tracks.append([(newTracksXVal, newTracksYVal)])
                    print "tracks appended to the end"

    def temporalRemove (self, tracks):
        lenTracks = [len(tr) for tr in tracks]

    def detectTrackCars(self, net):
        """Detect object classes in an image using pre-computed object proposals."""

        cap = cv2.VideoCapture("/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_10_2016_18_11_00_604698.h264")

        track_len = 10
        tracks = [] #stores center values of detected cars that pass the threshold
        numFrames = 0
        
        while (cap.isOpened()):
            ret, im = cap.read()
            frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_copy = im.copy()
            tracks = c.detectCars(im, im_copy, frame_gray, net, tracks, numFrames)

            cv2.polylines(im_copy, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

            print "before prev_gray"
            prev_gray = frame_gray

            cv2.imshow('frame', im_copy)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

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

    # print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    c.detectTrackCars (net)
    cv2.destroyAllWindows()
