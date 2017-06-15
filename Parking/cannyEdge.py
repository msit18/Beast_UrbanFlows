import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('/home/msit/Desktop/04-26_14.00.00_splice/005625.jpg')
crop = img[300:600, 0:1600]

img2 = cv2.imread('/home/msit/Desktop/04-26_17.00.00_splice/000225.jpg')
crop2 = img2[300:600, 0:1600]

spot = img2[300:600, 950:1400]

merge = img.copy()
merge[300:600, 950:1400] = img2[300:600, 950:1400]

#13:00
# analyzeImage = cv2.imread('001350.jpg')
#14:00
# analyzeImage = cv2.imread('../04-26_14.00.00_splice/000225.jpg')
#15:00
analyzeImage = cv2.imread('../04-26_15.00.00_splice/000000.jpg')

# array = [[ 247.84080505,  434.17144775,  626.03320312,  554.61505127],
# 		[  622.02294922,   405.94180298,  1028.23376465,   539.5322876 ],
# 		[ 1035.34484863,   379.25918579,  1355.61804199,   544.60784912]]

#13:00
# array = [[282, 414, 621, 551], [649, 400, 1016, 537], [1037, 375, 1350, 546]]

#14:00
# array = [[ 1019.9876709 ,   380.28192139,  1356.57019043,   550.44891357],
		 # [  619.75952148,   399.24371338,  1064.24182129,   540.62957764]]

#15:00
array = [[ 1045.41955566,   355.86233521,  1358.03820801,   545.76141357],
		 [ 1335.08007812,   438.89526367,  1480.54150391,   523.03808594],
		 [ 370.89178467,  407.24188232,  563.11383057,  564.81378174]]

def find(boxA, boxB):
	boxC = []
	print "There was an overlap"
	print "BoxA: ", boxA
	print "BoxB: ", boxB
	if boxA[0] < boxB[0]:
		boxC.append(boxA[0])
	else:
		boxC.append(boxB[0])

	if boxA[1] < boxB[1]:
		boxC.append(boxA[1])
	else:
		boxC.append(boxB[1])

	if boxA[2] > boxB[2]:
		boxC.append(boxA[2])
	else:
		boxC.append(boxB[2])

	if boxA[3] > boxB[3]:
		boxC.append(boxA[3])
	else:
		boxC.append(boxB[3])

 	print "BoxC: ", boxC
	# return the intersection over union value
	return boxC

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	boxC = []

	if iou > 0:
		boxC = find(boxA, boxB)
		# print "There was an overlap"
		# print "BoxA: ", boxA
		# print "BoxB: ", boxB
		# if boxA[0] < boxB[0]:
		# 	boxC.append(boxA[0])
		# else:
		# 	boxC.append(boxB[0])

		# if boxA[1] < boxB[1]:
		# 	boxC.append(boxA[1])
		# else:
		# 	boxC.append(boxB[1])

		# if boxA[2] > boxB[2]:
		# 	boxC.append(boxA[2])
		# else:
		# 	boxC.append(boxB[2])

		# if boxA[3] > boxB[3]:
		# 	boxC.append(boxA[3])
		# else:
		# 	boxC.append(boxB[3])
 
	 # 	print "BoxC: ", boxC
		# # return the intersection over union value
		return boxC
	else:
		print "There was no overlap"
		return 0

# x = 2
for x in range(len(array)):
	print "X: ", array[x][0]
# if x == 2:
	# cv2.rectangle(merge, (int(array[x][0]), int(array[x][1])), (int(array[x][2]), int(array[x][3])), (255, 255, 255), 3)
	smolSlice = analyzeImage[array[x][1]:array[x][3], array[x][0]:array[x][2]]
	gray1 = cv2.cvtColor(smolSlice,cv2.COLOR_BGR2GRAY)
	blur1 = cv2.blur(gray1,(20,20))
	blur3 = cv2.blur(gray1,(5,5))
	# cv2.imshow('smolSlice', smolSlice)
	cv2.imshow('smolblur1', blur1)

	cutSlice = merge[array[x][1]:array[x][3], array[x][0]:array[x][2]]
	gray2 = cv2.cvtColor(cutSlice,cv2.COLOR_BGR2GRAY)
	blur2 = cv2.blur(gray2,(20,20))
	blur4 = cv2.blur(gray2,(5,5))
	# cv2.imshow('cutSlice', cutSlice)
	# cv2.imshow('cutblur2', blur2)

	npblur2 = np.array(blur2)
	npblur1 = np.array(blur1)

	npdiff = npblur2-npblur1
	# npfilter = np.where(npdiff>230)
	# result = np.clip(npdiff, 30, 230)
	# result = np.fmin(npdiff, 240)

	# cv2.imshow('np', result)

	# difference = gray2-gray1
	difference = blur2-blur1
	difference2 = blur4-blur3
	# kernel = np.ones((3,3),np.uint8)
	# opening = cv2.morphologyEx(difference2,cv2.MORPH_OPEN,kernel, iterations = 4)

	blurCopy = np.zeros(npblur2.shape)
	blurCopy = npblur2
	print "Shape: ", npblur2.shape
	# print "blur11: ", blur1[1]
	# print "blur1: ", blur1[1][1]

	for y in range(npblur2.shape[0]):
		for x in range(npblur2.shape[1]):
			diff = blur2[y][x] - blur1[y][x]
			# print "Diff: ", diff
			if diff < 240 | diff > 10:
				blurCopy[y][x] = 0
			else:
				blurCopy[y][x] = diff

	# for x in range(len(blur2)):
	# 	print "x: ", x
	# 	diff = blur2[x]-blur1[x]
	# 	print "x: ", diff
	# 	print "x < 240: ", diff < 240
	# 	if diff < 240 | diff > 40:
	# 		blurCopy[x] = 0
	# 	else:
	# 		blurCopy[x] = diff

	cv2.imshow('blurCopy?', blurCopy)


	diff_copy = difference.copy()

# #######################################
# 	# Setup SimpleBlobDetector parameters.
# 	params = cv2.SimpleBlobDetector_Params()

# 	# params.filterByColor = True
# 	# params.blobColor = 255

# 	# Change thresholds
# 	params.minThreshold = 1000
# 	# params.maxThreshold = 100000

# 	# Filter by Area.
# 	params.filterByArea = True
# 	params.minArea = 150

# 	# # Filter by Circularity
# 	# params.filterByCircularity = True
# 	# params.minCircularity = 0.1

# 	# # Filter by Convexity
# 	# params.filterByConvexity = True
# 	# params.minConvexity = 0.87

# 	# Filter by Inertia
# 	# params.filterByInertia = True
# 	# params.minInertiaRatio = 0.01

# 	# Create a detector with the parameters
# 	detector = cv2.SimpleBlobDetector(params)


# 	# Detect blobs.
# 	keypoints = detector.detect(difference)
# 	print "Keypts: ", keypoints

# 	# Draw detected blobs as red circles.
# 	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# 	# the size of the circle corresponds to the size of blob
# 	color = cv2.cvtColor(diff_copy, cv2.COLOR_GRAY2BGR)
# 	im_with_keypoints = cv2.drawKeypoints(color, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 	# # Show blobs
# 	# cv2.imshow("Keypoints", im_with_keypoints)
# 	# cv2.waitKey(0)


	color = cv2.cvtColor(blurCopy, cv2.COLOR_GRAY2BGR)
	edges = cv2.Canny(blurCopy,500,500)

	# contours, hierarchy = cv2.findContours(diff_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# print "Contours: ", contours
	# print "Hiearchy: ", hierarchy

	# diff_copy = difference.copy()
	# cv2.drawContours(difference, contours, 0, (0,255,0), 6)

	# cnt = contours[6]
	# color = cv2.cvtColor(diff_copy, cv2.COLOR_GRAY2BGR)

	###Find the bbox of contour return
	bbox = []
	for total in range(len(contours)):
		cnt = contours[total]
		# print "cnt: ", cnt
		x,y,w,h = cv2.boundingRect(cnt)
		bbox.append([x, y, x+w, y+h])

		cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),2)

	print "Before sort: ", bbox
	# bbox = np.sort(bbox, axis=1)
	# print "bbox: ", bbox

	# print "bbox: ", bbox

	# rectList = cv2.groupRectangles(bbox, 1 , 0.1)
	# print "rectList: ", rectList

	# cv2.rectangle(color,(rectList[0][0][0], rectList[0][0][1]),(rectList[0][0][2], rectList[0][0][3]),(255,0, 0),2)
	# cv2.rectangle(color,(rectList[0][1][0], rectList[0][1][1]),(rectList[0][1][2], rectList[0][1][3]),(255,0, 0),2)
	# print "Rect0: ", rectList[0][0][0]
	# print "rect1: ", rectList[1]


	###Attempt to find the largest bbox/segmentation

	# largestArray = [color.shape[1]-1, color.shape[1], color.shape[0], color.shape[0]-1]
	biggestArea = [0, 0, 0, 0]
	area = 0
	for x in range(len(bbox)):
		# bboxcoord = [bbox[x][0], bbox[x][1], bbox[x][0]+bbox[x][2], bbox[x][1]+bbox[x][3]]
		print "bboxcoord: ", bbox[x]
		w = bbox[x][2] - bbox[x][0]
		print "w: ", w
		h = bbox[x][3] - bbox[x][1]
		print "h: ", h
		thisarea = w
		dist = math.sqrt(w**2 + h**2)
		print "this area: ", thisarea
		print "dist: ", dist
		# print "area of add: ", thisarea
		# largestArray = find(add, largestArray)

		# print "area: ", area
		if thisarea > area:
			area = thisarea
			biggestArea = bbox[x]

	if area == 0:
		print "nothing matched black image"



		# add = bb_intersection_over_union(bbox[x], bbox[x+1])
		# print "ADD: ", add
		# print "LargestArray so far: ", largestArray
		# if add != 0:
		# 	# bbox.append(add)
		# 	# np.append(bbox, add)
		# 	# print "bbox: ", bbox
		# 	thisarea = (add[2]-add[0])*(add[3]-add[1])
		# 	print "area of add: ", thisarea
		# 	largestArray = find(add, largestArray)

		# 	print "area: ", area
		# 	if thisarea > area:
		# 		area = thisarea
		# 		biggestArea = add

	# print "bbox: ", bbox
	print "area: ", area
	print "biggestArea: ", biggestArea
	# print "largestArray: ", largestArray

	# print "endbbox: ", bbox[-1][0]
	# cv2.rectangle(color,(largestArray[0], largestArray[1]),(largestArray[2], largestArray[3]),(0, 0, 255),2)
	cv2.rectangle(color,(biggestArea[0], biggestArea[1]),(biggestArea[2], biggestArea[3]),(0, 255, 255),2)

	print "Shape2: ", color.shape[0]
	print "Shape1: ", color.shape[1]
	print "----------------"

	# epsilon = 0.1*cv2.arcLength(cnt,True)
	# approx = cv2.approxPolyDP(cnt,epsilon,True)
	# print "approx: ", approx
	
	# cv2.rectangle(color,(0,0),(100,100),(0,255,0),2)

	# cv2.drawContours(diff_copy, [cnt], 0, (0,255,0), 3)

	# cv2.imshow('difference', difference)
	
	# cv2.imshow('difference2', difference2)
	cv2.imshow('diff_copy', color)








# cv2.imshow('show', crop)
# cv2.imshow('sho2', crop2)
# cv2.imshow('spot', spot)
# cv2.imshow('merge', merge)
# cv2.imshow('analyzeImage', analyzeImage)

# cv2.imwrite("emptySpots.jpg", merge)

# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)


# img = cv2.imread('001350.jpg',0)
# # edges = cv2.Canny(img,100,200)
# # edges = cv2.Canny(img,50,80)
# blur = cv2.blur(img,(5,5))
# # edges = cv2.Canny(blur,22,25)
# edges = cv2.Canny(blur, 10,20)
# ret, thresh = cv2.threshold(edges,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)

# array = [[ 247.84080505,  434.17144775,  626.03320312,  554.61505127],
# 		[  622.02294922,   405.94180298,  1028.23376465,   539.5322876 ],
# 		[ 1035.34484863,   379.25918579,  1355.61804199,   544.60784912]]

# for x in range(len(array)):
# 	cv2.rectangle(merge, (int(array[x][0]), int(array[x][1])), (int(array[x][2]), int(array[x][3])), (255, 255, 255), 3)

# img = cv2.imread('001350.jpg',0)

# crop1 = img[array[0][1]:array[0][3], array[0][0]:array[0][2]]
# blur = cv2.blur(crop1,(7,7))
# edges = cv2.Canny(blur, 10, 25)
# # edges = cv2.Canny(blur, 17, 25)
# # edges = cv2.Canny(blur, 15, 20)
# ret, thresh = cv2.threshold(edges,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# # cv2.imshow('gray', img)
# # cv2.imshow('crop1', crop1)
# cv2.imshow('edges', edges)

# crop2 = img[array[1][1]:array[1][3], array[1][0]:array[1][2]]
# blur2 = cv2.blur(crop2,(7,7))
# edges2 = cv2.Canny(blur2, 30, 50)
# # edges = cv2.Canny(blur, 17, 25)
# # edges = cv2.Canny(blur, 15, 20)
# ret, thresh = cv2.threshold(edges2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# cv2.imshow('edges2', edges2)


# smolblur = cv2.blur(img,(5,5))
# compedges = cv2.Canny(smolblur, 0, 30)
# # compedges = cv2.Canny(blur, 20, 25)
# ret, compthresh = cv2.threshold(compedges,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# cv2.imshow('compedges', compedges)


	cv2.waitKey(0)
	# cap.release()
	cv2.destroyAllWindows()
