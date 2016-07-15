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

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, time
import argparse

import video
from common import anorm2, draw_str
from time import clock

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


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

def demo(net):
    """Detect object classes in an image using pre-computed object proposals."""

    #cap = cv2.VideoCapture('/home/msit/py-faster-rcnn_BACK/data/demo/slavePi2_RW1600_RH1200_TT900_FR15_06_12_2016_17_34_39_833163.avi')
    #cap = cv2.VideoCapture('/home/msit/Presentation/OpenCV/vehicle_detection_haarcascades-master/slavePi4_RW1600_RH1200_TT180_FR15_06_03_2016_13_53_54_254155.h264')
    #cap = cv2.VideoCapture('/home/msit/Presentation/OpenCV/vehicle_detection_haarcascades-master/slavePi4_RW1600_RH1200_TT180_FR15_06_03_2016_13_53_54_254155.h264')
    cap = cv2.VideoCapture("/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_10_2016_18_11_00_604698.h264")

    track_len = 10
    tracks = [] #stores center values of detected cars that pass the threshold
    rectangleTracks = [] #coordinates of all the cars that are detected (x,y,w,h) Is not the four coordinates
    totalNumCars = 0
    totalNumCarsInFrame = 0
    
    try:
        while (cap.isOpened()):
            ret, im = cap.read()
            frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_copy = im.copy()

            if len(tracks) > 0:
                img0, img1 = prev_gray, frame_gray
                print "totalNumCarsInFrame: ", totalNumCarsInFrame
                print "tracks: ", tracks
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                print "p0: ", p0
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                print "p0r", p0r
                d = abs(p0-p0r).reshape(-1, 2).max(-1) #included for robustness - calculates diff btwn calc point and backtracks it
                good = d < 1 #returns a boolean value. This value is the threshold
                new_tracks = []
                for tr, (x, y), good_flag, (rectX, rectY, rectW, rectH) in zip(tracks, p1.reshape(-1, 2), good, rectangleTracks):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len: #remove point if it exceeds the track length
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(im_copy, (x, y), 2, (0, 255, 0), -1)

                tracks = new_tracks
                cv2.polylines(im_copy, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                #draw_str(im_copy, (20, 20), 'track count: %d' % len(tracks))
                #draw_str(im_copy, (40, 40), 'total car count: %d' % (totalNumCars))
                #draw_str(im_copy, (20, 60), 'car count: %d' % (totalNumCarsInFrame))

            # Detect all object classes and regress object bounds
            timer = Timer()
            timer.tic()
            scores, boxes = im_detect(net, im)
            timer.toc()
            # print ('Detection took {:.3f}s for '
            #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

            # Visualize detections for each class
            CONF_THRESH = 0.7
            NMS_THRESH = 0.3
            for cls_ind, cls in enumerate(CLASSES[1:]):
                #detect elements of interest here using rcnn
                cls_ind += 1 # because we skipped background
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
                cls_scores = scores[:, cls_ind]
                #All the elements that could potentially be the element of interest
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                #only keep elements which pass the threshold here
                dets = dets[keep, :]
                if cls == "car":
                    totalNumCarsInFrame = 0
                    #vis_detections(im, cls, dets, thresh=CONF_THRESH)
                    #method is vis_detections(im, class_name, dets, thresh=0.5)
                    inds = np.where(dets[:, -1] >= 0.5)[0] #Another threshold applied here, but not certain why
                    # if len(inds) == 0:
                    #     return

                    im = im[:, :, (2, 1, 0)]

                    #Calculate center of box, and draw on image
                    for i in inds:
                        bbox = dets[i, :4]
                        score = dets[i, -1]
                        x = int(bbox[0])
                        y = int(bbox[1])
                        w = (int(bbox[2]) - int(bbox[0]))
                        x1 = x + w
                        h = (int(bbox[3]) - int(bbox[1]))
                        y1 = y + h
                        cv2.rectangle(im_copy, (x,y), (x1, y1), (255, 0, 0), 3)
                        rectangleTracks.append([x,y,w,h])
                        totalNumCarsInFrame += 1
                        
                        car_centroid = [(x+(w/2)), (y+(h/2))]
                        cv2.circle (im_copy, ( (x+(w/2)) , (y+(h/2)) ), 4, (255, 0, 0), 4)
                        if car_centroid is not None:  
                            for a, b in np.float32(car_centroid).reshape(-1, 2):
                                tracks.append([(a, b)])

                    cv2.imshow('rectangles', im_copy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            prev_gray = frame_gray

            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     break

            print "take prev_gray frame"

    finally:
        end = time.time()
        print "end: ", end
        return end
    #cap.release()

def parse_args():
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

    args = parse_args()

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

    start = time.time()
    print "start: ", start
    end = demo (net)
    totalRunTime = end - start
    print "total run time was: ", totalRunTime
    cv2.destroyAllWindows()
