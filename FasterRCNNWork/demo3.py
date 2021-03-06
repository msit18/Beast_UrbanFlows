#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

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


# def vis_detections(im, class_name, dets, thresh=0.5):
#     """Draw detected bounding boxes."""
#     inds = np.where(dets[:, -1] >= thresh)[0]
#     if len(inds) == 0:
#         return

#     im = im[:, :, (2, 1, 0)]
#     fig, ax = plt.subplots(figsize=(12, 12))
#     #substitute this imshow with openCV visualization libraries
#     while True:
#         cv2.imshow('image', im)
#         ch = 0xFF & cv2.waitkey(1)
#         if ch == 27:
#             break

#     #ax.imshow(im, aspect='equal')
#     for i in inds:
#         bbox = dets[i, :4]
#         score = dets[i, -1]

# #substitute this part here for openCV visualization libraries
#         ax.add_patch(
#             plt.Rectangle((bbox[0], bbox[1]),
#                           bbox[2] - bbox[0],
#                           bbox[3] - bbox[1], fill=False,
#                           edgecolor='red', linewidth=3.5)
#             )
#         ax.text(bbox[0], bbox[1] - 2,
#                 '{:s} {:.3f}'.format(class_name, score),
#                 bbox=dict(facecolor='blue', alpha=0.5),
#                 fontsize=14, color='white')
        
#         #put in analysis here. Using bbox, if the box is similar/close enough to
#         #a previous box, then it does not get incremented. Otherwise, +1 to car count.
#         if class_name == "car":
#             print "found a car"
#             #plt.show()

#     ax.set_title(('{} detections with '
#                   'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                   thresh),
#                   fontsize=14)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.draw()
    
#     plt.show()

def demo(net):
    """Detect object classes in an image using pre-computed object proposals."""

    cap = cv2.VideoCapture('/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_10_2016_18_11_00_604698.avi')

    print "before the while loop"
    while (cap.isOpened()):
        ret, im = cap.read()
        im_copy = im.copy()

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, im)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class
        CONF_THRESH = 0.7
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            if cls == "car":
                #vis_detections(im, cls, dets, thresh=CONF_THRESH)
                #method is vis_detections(im, class_name, dets, thresh=0.5)
                inds = np.where(dets[:, -1] >= 0.5)[0]
                if len(inds) == 0:
                    return

                im = im[:, :, (2, 1, 0)]

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    x = int(bbox[0])
                    y = int(bbox[1])
                    x1 = x + (int(bbox[2]) - int(bbox[0]))
                    y1 = y + (int(bbox[3]) - int(bbox[1]))
                    cv2.rectangle(im_copy, (x,y), (x1, y1), (255, 0, 0), 3)
                    # if cls == "car":
                    #     print "found a car"

            cv2.imshow('rectangles', im_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    #cv2.destroyAllWindows()

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

    demo (net)