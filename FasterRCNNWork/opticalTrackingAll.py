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
from matplotlib import path
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

    def __init__(self):
        self.totalCarCount = 0
        self.carCountMin = 0

        self.totalBusCount = 0
        self.totalBikeCount = 0
        self.totalPeopleCount = 0

    def detectionMethod(self, im, im_copy, frame_gray, net, p, detectedItem, colorScheme):
        scores, boxes = im_detect(net, im)
        NMS_THRESH = 0.3
        detectedItemsInThisFrame = []

        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32) #stacks them together
            keep = nms(dets, NMS_THRESH) #Removes overlapping bounding boxes
            dets = dets[keep, :]

            #print "if cls == {0}: {1}".format(str(detectedItem), (cls == detectedItem))
            if cls == detectedItem:
                print "under else"
                inds = np.where(dets[:, -1] >= 0.5)[0] #Threshold applied to score values here
                im = im[:, :, (2, 1, 0)]

                for i in inds:
                    print "running for loop"
                    bbox = dets[i, :4]
                    detectedBBox = bbox.astype(int)
                    score = dets[i, -1]
                    bboxCentroid = c.mathArrayCentroid(detectedBBox)
                    cv2.circle(im_copy, bboxCentroid, 5, colorScheme, -1) #bbox centroid of detectedItemArray
                    #Calculate bbox centroid. Use it to determine if the item should be added to detectedItemArray

                    #Check if the centroid of the detected box is within the designated traffic intersection area
                    if (str(detectedItem) == "car") or (str(detectedItem) == "bus"):
                        print "if statement, detected car or bus"
                        if p.contains_point(bboxCentroid) == 1:
                            print "within area"
                            #Calculate corners of interest within the bounding box area and add them all to the corner array
                            detectedPixels = frame_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]] #[y1:y2, x1:x2]
                            detectedPixelsColor = im_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]] #for show on colored image
                            corners = cv2.goodFeaturesToTrack(detectedPixels, mask=detectedPixels, **self.feature_params).reshape(-1, 2)

                            # for x, y in np.float32(corners).reshape(-1, 2): #black
                            #     cv2.circle(detectedPixels, (x,y), 5, (0, 0, 0), -1)
                            #     cv2.circle(detectedPixelsColor, (x, y), 5, (0, 0, 0), -1)

                            detectedItemsInThisFrame.append([[detectedBBox, corners]])
                        else:
                            print "car/bus not added. Coordinates: ", bbox
                    
                    else:
                        print "else not car or bus detected"
                        detectedPixels = frame_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]] #[y1:y2, x1:x2]
                        detectedPixelsColor = im_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]] #for show on colored image
                        corners = cv2.goodFeaturesToTrack(detectedPixels, mask=detectedPixels, **self.feature_params).reshape(-1, 2)

                print "detectedItemsInThisFrame len: {0}-------------------------------------".format(len(detectedItemsInThisFrame))
                print "detectedItemsInThisFrame: ", detectedItemsInThisFrame

                return detectedItemsInThisFrame

#Detects cars within frame, detects corners of interest within detected car frame, outputs array
    def carDetectionMethodAll(self, im, im_copy, frame_gray, net, detectedCarsInThisFrame, p):
        scores, boxes = im_detect(net, im)

        # Visualize detections for each class
        # CONF_THRESH = 0.7
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
            if cls == "person":
                indsPPL = np.where(dets[:, -1] >= 0.05)[0] #Threshold applied to score values here

                im = im[:, :, (2, 1, 0)]

                for i in indsPPL:
                    bboxPPL = dets[i, :4]
                    pplDetectedBBox = bboxPPL.astype(int)
                    score = dets[i, -1]
                    bboxPPLCentroid = c.mathArrayCentroid(pplDetectedBBox)
                    cv2.circle(im_copy, bboxPPLCentroid, 5, (190, 86, 252), -1) #pink, bbox for people
                    cv2.rectangle(im_copy, (pplDetectedBBox[0], pplDetectedBBox[1]), (pplDetectedBBox[2], pplDetectedBBox[3]), (190, 86, 252), 3) #pink for people

            elif cls == "bus":
                indsBUS = np.where(dets[:, -1] >= 0.5)[0] #Threshold applied to score values here
                im = im[:, :, (2, 1, 0)]

                for i in indsBUS:
                    bboxBUS = dets[i, :4]
                    busDetectedBBox = bboxBUS.astype(int)
                    score = dets[i, -1]
                    bboxBUSCentroid = c.mathArrayCentroid(busDetectedBBox)
                    cv2.circle(im_copy, bboxBUSCentroid, 5, (0, 0, 0), -1)# black, bbox for bus
                    cv2.rectangle(im_copy, (busDetectedBBox[0], busDetectedBBox[1]), (busDetectedBBox[2], busDetectedBBox[3]), (0, 0, 0), 3) #black for bus

            elif cls == "bicycle":
                indsBIKE = np.where(dets[:, -1] >= 0.25)[0]
                im = im[:, :, (2, 1, 0)]

                for i in indsBIKE:
                    bboxBIKE = dets[i, :4]
                    bikeDetectedBBox = bboxBIKE.astype(int)
                    score = dets[i, -1]
                    bboxBIKECentroid = c.mathArrayCentroid(bikeDetectedBBox)
                    cv2.circle(im_copy, bboxBIKECentroid, 5, (0, 0, 0), -1)# black, bbox for bus
                    cv2.rectangle(im_copy, (bikeDetectedBBox[0], bikeDetectedBBox[1]), (bikeDetectedBBox[2], bikeDetectedBBox[3]), (0, 144, 255), 3) #gold for bike

            if cls == "car":
                inds = np.where(dets[:, -1] >= 0.5)[0] #Threshold applied to score values here
                im = im[:, :, (2, 1, 0)]

                #Calculate center of box, and draw on image. Use cvDrawBBox for cv2 (expects integers)
                #x = bbox[0], y = bbox[1] (top left corner)
                #x1 = bbox[2], y1 = bbox[3] (bottom right corner)
                #print "Number of cars in frame: ", len(inds)
                for i in inds:
                    bbox = dets[i, :4]
                    carDetectedBBox = bbox.astype(int)
                    score = dets[i, -1]
                    bboxCentroid = c.mathArrayCentroid(carDetectedBBox)
                    cv2.circle(im_copy, bboxCentroid, 5, (255, 255, 0), -1) #light blue, bbox centroid of detectedCars
                    #Calculate bbox centroid. Use it to determine if the car should be added to detectedCars

                    #Check if the centroid of the detected box is within the designated traffic intersection area
                    if p.contains_point(bboxCentroid) == 1:
                        print "within area"
                        #Calculate corners of interest within the bounding box area and add them all to the carCorner array
                        detectedCarPixels = frame_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]] #[y1:y2, x1:x2]
                        detectedCarPixelsColor = im_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]] #for show on colored image
                        carCorners = cv2.goodFeaturesToTrack(detectedCarPixels, mask=detectedCarPixels, **self.feature_params).reshape(-1, 2)

                        # for x, y in np.float32(carCorners).reshape(-1, 2): #black
                        #     cv2.circle(detectedCarPixels, (x,y), 5, (0, 0, 0), -1)
                        #     cv2.circle(detectedCarPixelsColor, (x, y), 5, (0, 0, 0), -1)

                        detectedCarsInThisFrame.append([[carDetectedBBox, carCorners]])

                    else:
                        print "car not added. Coordinates: ", bbox

                print "detectedCarsInThisFrame len: {0}-------------------------------------".format(len(detectedCarsInThisFrame))
                print "detectedCarsInThisFrame: ", detectedCarsInThisFrame

                return detectedCarsInThisFrame

#TO FIX: NEED A BETTER WAY TO COUNT CARS. CURRENT METHOD IS A LITTLE FAULTY, BUT CONSERVATIVE (CANCELS OUT THE FALSE POSITIVES IN THE END)
#ONCE ALL THE PAST DETECTED CARS ARE MATCHED, NO NEW CARS CAN BE ADDED (THEY SHOULD BE TAKEN INTO ACCOUNT ACTUALLY... NEED TO CHECK)
    # def thresholding (self, detectedCars, newCars, im_copy, frameNum):
    #     aggregatedCars = []
    #     print "detectedCars len: ", len(detectedCars)
    #     print "newCars len: ", len(newCars)
    #     for singleNewCarIndex in range(len(newCars)):
    #         for singleDetectedCarIndex in range(len(detectedCars)):
    #             confirmAppended = False

    #             #Weighted centroid from last detected Frame
    #             print "detectedCars[singleDetectedCarIndex][-1][0][0]: ", detectedCars[singleDetectedCarIndex][-1][0]
    #             print "newCars[singleNewCarIndex][-1][0][0]: ", newCars[singleNewCarIndex][-1][0]

    #             weightedDetectedCarsCornerCentroid = c.calcWeightedFix(detectedCars[singleDetectedCarIndex])
    #             print "weightedDetectedCarsCornerCentroid: ", weightedDetectedCarsCornerCentroid
    #             cv2.circle(im_copy, weightedDetectedCarsCornerCentroid, 5, (0, 37, 255), -1) #bright red

    #             weightedNewCarsCornerCentroid = c.calcWeightedFix(newCars[singleNewCarIndex])
    #             print "weightedNewCarsCornerCentroid: ", weightedNewCarsCornerCentroid
    #             cv2.circle(im_copy, weightedNewCarsCornerCentroid, 5, (0, 84, 255), -1) #orange, newly detected car

    #             print "math for wcenter: ", math.hypot(abs(weightedNewCarsCornerCentroid[0] - weightedDetectedCarsCornerCentroid[0]), abs(weightedNewCarsCornerCentroid[1] - weightedDetectedCarsCornerCentroid[1]))
    #             if math.hypot(abs(weightedNewCarsCornerCentroid[0] - weightedDetectedCarsCornerCentroid[0]), abs(weightedNewCarsCornerCentroid[1] - weightedDetectedCarsCornerCentroid[1])) < 115:
    #                 print "Less than 100**********"
    #                 print "DRAW PURPLE RECTANGLE"
    #                 detectedCars[singleDetectedCarIndex].append(newCars[singleNewCarIndex][0])
    #                 aggregatedCars.append(detectedCars[singleDetectedCarIndex])
    #                 cv2.circle(im_copy, c.mathArrayCentroid(detectedCars[singleDetectedCarIndex][-1][0]), 5, (150, 50, 100), -1) #purple
    #                 cv2.rectangle(im_copy, (detectedCars[singleDetectedCarIndex][-1][0][0], detectedCars[singleDetectedCarIndex][-1][0][1]), (detectedCars[singleDetectedCarIndex][-1][0][2], detectedCars[singleDetectedCarIndex][-1][0][3]), (150, 50, 100), 3) #purple
    #                 del detectedCars[singleDetectedCarIndex]
    #                 confirmAppended = True
    #                 break

    #         if (confirmAppended == False):
    #             print "DRAW GREEN RECTANGLE"
    #             aggregatedCars.append(newCars[singleNewCarIndex])
    #             cv2.circle(im_copy, c.mathArrayCentroid(newCars[singleNewCarIndex][-1][0]), 5, (0, 255, 0), -1) #green
    #             cv2.rectangle(im_copy, (newCars[singleNewCarIndex][-1][0][0], newCars[singleNewCarIndex][-1][0][1]), (newCars[singleNewCarIndex][-1][0][2], newCars[singleNewCarIndex][-1][0][3]), (0, 255, 0), 3) #green
    #             self.totalCarCount += 1
    #             self.carCountMin += 1
    #             print "If statement is true. += totalCarCount. Appended size is now: ", len(aggregatedCars)
    #         print "NEXT CAR;;;;;;;;;;;;;;;;;;;;;;;;;;;;"

    #     if (len(detectedCars)>0 and (frameNum%10>0)):
    #         print "adding remaining cars to the end of aggregatedCars"
    #         for remainingCar in detectedCars:
    #             aggregatedCars.append(remainingCar)
    #             print "DRAW BLUE RECTANGLE"
    #             cv2.circle(im_copy, c.mathArrayCentroid(remainingCar[-1][0]), 5, (255,0,0), -1) #blue
    #             cv2.rectangle(im_copy, (remainingCar[-1][0][0], remainingCar[-1][0][1]), (remainingCar[-1][0][2], remainingCar[-1][0][3]), (255, 0, 0), 3) #blue

    #     print "-----------------------len aggregatedCars: ", len(aggregatedCars)

    #     return aggregatedCars

    def thresholdingGeneral(self, detectedItemArray, newItemArray, im_copy, frameNum, itemDetecting, itemDetectingTotal):
        aggregatedItems = []
        print "detectedItemArray len: ", len(detectedItemArray)
        print "newItemArray len: ", len(newItemArray)
        for singleNewItemArrayIndex in range(len(newItemArray)):
            for singleDetectedItemIndex in range(len(detectedItemArray)):
                confirmAppended = False

                #Weighted centroid from last detected Frame
                print "detectedItemArray[singleDetectedItemIndex][-1][0][0]: ", detectedItemArray[singleDetectedItemIndex][-1][0]
                print "newItemArray[singleNewItemArrayIndex][-1][0][0]: ", newItemArray[singleNewItemArrayIndex][-1][0]

                weightedDetectedItemCornerCentroid = c.calcWeightedFix(detectedItemArray[singleDetectedItemIndex])
                print "weightedDetectedItemCornerCentroid: ", weightedDetectedItemCornerCentroid
                cv2.circle(im_copy, weightedDetectedItemCornerCentroid, 5, (0, 37, 255), -1) #bright red

                weightedNewItemCornerCentroid = c.calcWeightedFix(newItemArray[singleNewItemArrayIndex])
                print "weightedNewItemCornerCentroid: ", weightedNewItemCornerCentroid
                cv2.circle(im_copy, weightedNewItemCornerCentroid, 5, (0, 84, 255), -1) #orange, newly detected car

                print "math for wcenter: ", math.hypot(abs(weightedNewItemCornerCentroid[0] - weightedDetectedItemCornerCentroid[0]), abs(weightedNewItemCornerCentroid[1] - weightedDetectedItemCornerCentroid[1]))
                if math.hypot(abs(weightedNewItemCornerCentroid[0] - weightedDetectedItemCornerCentroid[0]), abs(weightedNewItemCornerCentroid[1] - weightedDetectedItemCornerCentroid[1])) < 115:
                    print "Less than 100**********"
                    print "DRAW PURPLE RECTANGLE"
                    detectedItemArray[singleDetectedItemIndex].append(newItemArray[singleNewItemArrayIndex][0])
                    aggregatedItems.append(detectedItemArray[singleDetectedItemIndex])
                    cv2.circle(im_copy, c.mathArrayCentroid(detectedItemArray[singleDetectedItemIndex][-1][0]), 5, (150, 50, 100), -1) #purple
                    cv2.rectangle(im_copy, (detectedItemArray[singleDetectedItemIndex][-1][0][0], detectedItemArray[singleDetectedItemIndex][-1][0][1]), (detectedItemArray[singleDetectedItemIndex][-1][0][2], detectedItemArray[singleDetectedItemIndex][-1][0][3]), (150, 50, 100), 3) #purple
                    del detectedItemArray[singleDetectedItemIndex]
                    confirmAppended = True
                    break

            if (confirmAppended == False):
                print "DRAW GREEN RECTANGLE"
                aggregatedItems.append(newItemArray[singleNewItemArrayIndex])
                cv2.circle(im_copy, c.mathArrayCentroid(newItemArray[singleNewItemArrayIndex][-1][0]), 5, (0, 255, 0), -1) #green
                cv2.rectangle(im_copy, (newItemArray[singleNewItemArrayIndex][-1][0][0], newItemArray[singleNewItemArrayIndex][-1][0][1]), (newItemArray[singleNewItemArrayIndex][-1][0][2], newItemArray[singleNewItemArrayIndex][-1][0][3]), (0, 255, 0), 3) #green
                if itemDetecting == "car":
                    self.totalCarCount += 1
                    self.carCountMin += 1
                else:
                    itemDetectingTotal += 1
                print "If statement is true. += totalItemCount. Appended size is now: ", len(aggregatedItems)
            print "NEXT CAR;;;;;;;;;;;;;;;;;;;;;;;;;;;;"

        if (len(detectedItemArray)>0 and (frameNum%10>0)):
            print "adding remaining cars to the end of aggregatedItems"
            for remainingItem in detectedItemArray:
                aggregatedItems.append(remainingItem)
                print "DRAW BLUE RECTANGLE"
                cv2.circle(im_copy, c.mathArrayCentroid(remainingItem[-1][0]), 5, (255,0,0), -1) #blue
                cv2.rectangle(im_copy, (remainingItem[-1][0][0], remainingItem[-1][0][1]), (remainingItem[-1][0][2], remainingItem[-1][0][3]), (255, 0, 0), 3) #blue

        print "-----------------------len aggregatedItems: ", len(aggregatedItems)

        return aggregatedItems

##Helper Methods
    def calcWeightedFix(self, inputArray):
        IACornerPts = inputArray[-1][1]
        xtot = 0
        ytot = 0
        for xcp, ycp in IACornerPts:
            xtot += xcp
            ytot += ycp
        xIMCOPY2 = inputArray[-1][0][0] + xtot/len(IACornerPts)
        yIMCOPY2 = inputArray[-1][0][1] + ytot/len(IACornerPts)
        return (int(xIMCOPY2), int(yIMCOPY2))

    def mathArrayCentroid (self, givenBBox):
        return (int(givenBBox[0]+(givenBBox[2]-givenBBox[0])/2), int(givenBBox[1]+(givenBBox[3]-givenBBox[1])/2))

    # def visualizeTrackedCars (self, allCars, im_copy):
    #     bboxCoordinates = [coordArrays[-1][0] for coordArrays in allCars]
    #     for car in allCars:
    #         centers = []
    #         for pts in car:
    #             centers.append([c.mathArrayCentroid(pts[0])])
    #         cv2.polylines(im_copy, np.array([centers], dtype=np.int32), False, (255, 50, 125)) #purple to follow tracked box

    def visualizeTrackedItem (self, allitems, im_copy):
        bboxCoordinates = [coordArrays[-1][0] for coordArrays in allItems]
        for item in allItems:
            centers = []
            for pts in item:
                centers.append([c.mathArrayCentroid(pts[0])])
            cv2.polylines(im_copy, np.array([centers], dtype=np.int32), False, (255, 50, 125)) #purple to follow tracked box

    def analyzeItem (self, itemName, detectedItemArray, colorCenter, itemDetectingTotal, im_copy, frame_gray, p):
        newItemArray = []
        if len(detectedItemArray) <= 0:
            detectedItemArray = c.detectionMethod(im, im_copy, frame_gray, net, p, str(itemName), colorCenter)
            print "len(detectedItemArray): ", len(detectedItemArray)
            if itemName == "car":
                self.totalCarCount += len(detectedItemArray)
                self.carCountMin += len(detectedItemArray)
            else:
                itemDetectingTotal += len(detectedItemArray)

        elif len(detectedCars) > 0:
            print "detectedItemArray len: ", len(detectedItemArray)
            newItemArray = c.detectionMethod(im, im_copy, frame_gray, net, p, str(itemName), colorCenter)
            print "Updated newItemArray value: ", len(newItemArray)
            detectedItemArray = c.thresholdingGeneral(detectedItemArray, newItemArray, im_copy, frameNum, str(itemName), itemDetectingTotal)
        
            c.visualizeTrackedItem(detectedItemArray, im_copy)

        return detectedItemArray

    def mainAll(self, net, im):
        """Detect object classes in an image using pre-computed object proposals."""
        cap = cv2.VideoCapture("/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_10_2016_18_11_00_604698.h264")
        carCountFile = open('carCountFile2.txt', 'w')

        #TO FIX? CREATE AREA CLOSER TO THE BOTTOM OF THE SCREEN THAT ALLOWS FOR GREATER THRESHOLD DISTANCE. CARS ARE TRAVELING TOO FAST IN THAT AREA
        detectedCars = [] #stores all information about the detected cars
        detectedPeople = []
        detectedBuses = []
        detectedBikes = []

        frameNum = 0
        p = path.Path([(0, 362), (741, 229), (1581, 390), (1597, 704), (450, 1017), (0, 785)])
        bboxIntersection = np.array([[0, 362], [741, 229], [1581, 390], [1597, 704], [450, 1017], [0, 785]])
        
        startTime = time.time()

        while (cap.isOpened()):
            ret, im = cap.read()
            frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_copy = im.copy()
            cv2.polylines(im_copy, np.int32([bboxIntersection]), False, (0,255,0))

            detectedCars = c.analyzeItem("car", detectedCars, (255, 255, 0), self.totalCarCount, im_copy, frame_gray, p) #light blue
            detectedPeople = c.analyzeItem("person", detectedPeople, (190, 86, 252), self.totalPeopleCount, im_copy, frame_gray, p) #pink
            detectedBuses = c.analyzeItem("bus", detectedBuses, (0, 0, 0), self.totalBusCount, im_copy, frame_gray, p) #black
            detectedBikes = c.analyzeItem("bicycle", detectedBikes, (0, 144, 255), self.totalBikeCount, im_copy, frame_gray, p) #orange

            # if len(detectedCars) <= 0:
            #     detectedCars = c.detectionMethod(im, im_copy, frame_gray, net, p, "car")
            #     print "len(detectedCars): ", len(detectedCars)
            #     self.totalCarCount += len(detectedCars)
            #     self.carCountMin += len(detectedCars)
            # elif len(detectedCars) > 0:
            #     print "detectedcars len: ", len(detectedCars)
            #     newCars = c.detectionMethod(im, im_copy, frame_gray, net, p, "car")
            #     print "Updated newCars value: ", len(newCars)
            #     detectedCars = c.thresholding(detectedCars, newCars, im_copy, frameNum)
            
            #     c.visualizeTrackedCars(detectedCars, im_copy)

            # elif len(detectedBuses) <= 0:
            #     detectedBuses = c.detectionMethod(im, im_copy, frame_gray, net, p, "bus")
            #     print "len(detectedBuses): ", len(detectedBuses)
            #     self.totalBusCount += len(detectedBuses)
            # elif len(detectedBuses) > 0:
            #     print "detectedBuses len: ", len(detectedBuses)
            #     newBuses = c.detectionMethod(im, im_copy, frame_gray, net, p, "bus")
            #     print "Updated newBuses value: ", len(newBuses)
            #     detectedBuses = c.thresholding(detectedBuses, newBuses, im_copy, frameNum)

            #     c.visualizeTrackedCars(detectedBuses, im_copy)

            # elif len(detectedBikes) <= 0:
            #     detectedBikes = c.detectionMethod(im, im_copy, frame_gray, net, p, "bicycle")
            #     print "len(detectedBikes): ", len(detectedBikes)
            #     self.totalBikeCount += len(detectedBikes)
            # elif len(detectedBikes) > 0:
            #     print "detectedBikes len: ", len(detectedBikes)
            #     newBikes = c.detectionMethod(im, im_copy, frame_gray, net, p, "bicycle")
            #     print "Updated bike value: ", len(newBikes)
            #     detectedBikes = c.thresholding(detectedBikes, newBikes, im_copy, frameNum)

            # elif len(detectedPeople) <= 0:
            #     detectedPeople = c.detectionMethod(im, im_copy, frame_gray, net, p, "person")
            #     print "len(detectedPeople): ", len(detectedPeople)
            #     self.totalPeopleCount += 0
            # elif len(detectedPeople) > 0:
            #     print "detectedPeople len: ", len(detectedPeople)
            #     newPeople = c.detectionMethod(im, im_copy, frame_gray, net, p, "person")
            #     print "Updated people value: ", len(newPeople)
            #     detectedPeople = c.thresholding(detectedPeople, newPeople, im_copy, frameNum)
            
            frameNum += 1
            print "====================================FRAMENUM {0}, TOTAL CAR COUNTS {1}. CAR COUNT MIN {2}. TOTAL BUS COUNT: {3}, TOTAL PEOPLE COUNT: {4}, TOTAL BIKE COUNT: {5}"\
                    .format(frameNum, self.totalCarCount, self.carCountMin, self.totalBusCount, self.totalPeopleCount, self.totalBikeCount)

            if (frameNum%150==0):
                print "resetting carCountMin!!!!!"
                print "carCountMin: ", self.carCountMin
                carCountFile.write('approxTimeSec: {0}\tCarCountMin: {1}\ttotalCarCount: {2}\ttotalBusCount: {3}\ttotalPeopleCount: {4}\ttotalBikeCount: {5}\n'.\
                                    format((frameNum/15), self.carCountMin, self.totalCarCount, self.totalBusCount, self.totalPeopleCount, self.totalBikeCount))
                self.carCountMin = 0
                print "carCountMin: ", self.carCountMin

            prev_gray = frame_gray

            # cv2.imshow('frame', im_copy)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            print "take prev_gray frame****************************************************"
        
        print "totalCarCount: ", self.totalCarCount
        print "carCountMin last: ", self.carCountMin
        carCountFile.write('approxTimeSec: {0}\tCarCount: {1}\tTotalCarCount: {2}\ttotalBusCount: {3}\ttotalPeopleCount: {4}\ttotalBikeCount: {5}\n'\
                            .format((frameNum/15), self.carCountMin, self.totalCarCount, self.totalBusCount, self.totalPeopleCount, self.totalBikeCount))

        endTime = time.time()
        totalCalcTime = endTime -startTime
        carCountFile.write('numFrames: {0}\tCarCount: {1}\tTotalCarCount: {2}\ttotalBusCount: {3}\ttotalPeopleCount: {4}\ttotalBikeCount: {5}\n'\
                            .format(frameNum, self.carCountMin, self.totalCarCount, self.totalBusCount, self.totalPeopleCount, self.totalBikeCount))
        carCountFile.write("totalRunning Time: {0}\n".format(totalCalTime))
        print "totalRunning Time: ", totalCalcTime

        carCountFile.close()
        cap.release()

    # def main(self, net):
    #     """Detect object classes in an image using pre-computed object proposals."""

    #     cap = cv2.VideoCapture("/media/senseable-beast/beast-brain-1/Data/TrafficIntersectionVideos/slavePi2_RW1600_RH1200_TT900_FR15_06_10_2016_18_11_00_604698.h264")

    #     #TO FIX? CREATE AREA CLOSER TO THE BOTTOM OF THE SCREEN THAT ALLOWS FOR GREATER THRESHOLD DISTANCE. CARS ARE TRAVELING TOO FAST IN THAT AREA
    #     detectedCars = [] #stores all information about the detected cars
    #     newCars = [] #newly detectedCars. Matched to ones stored in detectedCars in thresholding method
    #     frameNum = 0
    #     p = path.Path([(0, 362), (741, 229), (1581, 390), (1597, 704), (450, 1017), (0, 785)])
    #     bboxIntersection = np.array([[0, 362], [741, 229], [1581, 390], [1597, 704], [450, 1017], [0, 785]])
    #     carCountFile = open('carCountFile.txt', 'w')
    #     startTime = time.time()

    #     while (cap.isOpened()):
    #         ret, im = cap.read()
    #         frame_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #         im_copy = im.copy()
    #         cv2.polylines(im_copy, np.int32([bboxIntersection]), False, (0,255,0))

    #         if len(detectedCars) <= 0:
    #             detectedCars = c.carDetectionMethodAll(im, im_copy, frame_gray, net, detectedCars, p)
    #             print "len(detectedCars): ", len(detectedCars)
    #             self.totalCarCount += len(detectedCars)
    #             self.carCountMin += len(detectedCars)
    #         elif len(detectedCars) > 0:
    #             print "detectedcars len: ", len(detectedCars)
    #             newCars = c.carDetectionMethodAll(im, im_copy, frame_gray, net, detectedCars, p)
    #             print "Updated newCars value: ", len(newCars)
    #             detectedCars = c.thresholding(detectedCars, newCars, im_copy, frameNum)
            
    #         c.visualizeTrackedCars(detectedCars, im_copy)
    #         frameNum += 1
    #         print "====================================================FRAMENUM {0}, TOTAL CAR COUNTS {1}. CAR COUNT MIN {2}".format(frameNum, self.totalCarCount, self.carCountMin)

    #         if (frameNum%150==0):
    #             print "resetting carCountMin!!!!!"
    #             print "carCountMin: ", self.carCountMin
    #             carCountFile.write('approxTimeSec: {0}\tCarCount: {1}\ttotalCarCount: {2}\n'.format((frameNum/15), self.carCountMin, self.totalCarCount))
    #             self.carCountMin = 0
    #             print "carCountMin: ", self.carCountMin

    #         prev_gray = frame_gray

    #         cv2.imshow('frame', im_copy)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #         print "take prev_gray frame****************************************************"
        
    #     print "totalCarCount: ", self.totalCarCount
    #     print "carCountMin last: ", self.carCountMin
    #     carCountFile.write('approxTimeSec: {0]\tCarCount: {1}\tTotalCarCount: {2}\n'.format((frameNum/15), self.carCountMin, self.totalCarCount))

    #     endTime = time.time()
    #     totalCalcTime = endTime -startTime
    #     carCountFile.write('totalCalTime: {0}\tCarCount: {1}\tTotalCarCount: {2}\n'.format(frameNum, self.carCountMin, self.totalCarCount))
    #     print "totalRunning Time: ", totalCalcTime

    #     carCountFile.close()
    #     cap.release()


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

    #c.main(net)
    c.mainAll(net, im)
    cv2.destroyAllWindows()
