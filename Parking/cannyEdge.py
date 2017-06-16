import cv2
import numpy as np
from matplotlib import pyplot as plt
import math, time

img = cv2.imread('/home/msit/Desktop/04-26_14.00.00_splice/005625.jpg')
crop = img[300:600, 0:1600]
img2 = cv2.imread('/home/msit/Desktop/04-26_17.00.00_splice/000225.jpg')
crop2 = img2[300:600, 0:1600]

spot = img2[300:600, 950:1400]

merge = img.copy()
merge[300:600, 950:1400] = img2[300:600, 950:1400]

##13:00
# analyzeImage = cv2.imread('001350.jpg')
##14:00
# analyzeImage = cv2.imread('../04-26_14.00.00_splice/000225.jpg')
## 15:00
analyzeImage = cv2.imread('../04-26_15.00.00_splice/000000.jpg')



## 13:00
# array = [[282, 414, 621, 551], [649, 400, 1016, 537], [1037, 375, 1350, 546]]
## 14:00
# array = [[ 1019.9876709 ,   380.28192139,  1356.57019043,   550.44891357],
		 # [  619.75952148,   399.24371338,  1064.24182129,   540.62957764]]
##15:00
array = [[ 1045.41955566,   355.86233521,  1358.03820801,   545.76141357],
		 [ 1335.08007812,   438.89526367,  1480.54150391,   523.03808594],
		 [ 370.89178467,  407.24188232,  563.11383057,  564.81378174]]

def find(boxA, boxB):
	boxC = []
	print "There was an overlap"
	print "BoxA: ", boxA
	print "BoxB: ", boxB
	boxC.append(min(boxA[0], boxB[0]))
	boxC.append(min(boxA[1], boxB[1]))
	boxC.append(max(boxA[2], boxB[2]))
	boxC.append(max(boxA[3], boxB[3]))
 	print "BoxC: ", boxC
	return boxC

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	if iou > 0:
		return find(boxA, boxB)
	else:
		print "There was no overlap"
		return 0


# x = 0
# if x == 0:
for x in range(len(array)):
	###Image slicing
	smolSlice = analyzeImage[array[x][1]:array[x][3], array[x][0]:array[x][2]]
	gray1 = cv2.cvtColor(smolSlice,cv2.COLOR_BGR2GRAY)
	blur1 = cv2.blur(gray1,(20,20))
	# cv2.imshow('smolSlice', smolSlice)
	# cv2.imshow('smolblur1', blur1) ###############################

	cutSlice = merge[array[x][1]:array[x][3], array[x][0]:array[x][2]]
	gray2 = cv2.cvtColor(cutSlice,cv2.COLOR_BGR2GRAY)
	blur2 = cv2.blur(gray2,(20,20))
	# cv2.imshow('cutSlice', cutSlice)
	# cv2.imshow('cutblur2_early', blur2)##################################

	difference = blur2-blur1 #used as a baseline here.
	# cv2.imshow('difference', difference)


	###Blurring and finding edges
	npblur2 = np.asarray(blur2)
	npblur1 = np.asarray(blur1)
	npdiff = npblur2-npblur1
	# cv2.imshow('npdiff', npdiff)

	newArray = np.where(npdiff < 240, 0, npdiff)
	finalArray = np.where(newArray < 10, 0, newArray)
	# cv2.imshow('newArray', newArray)######################################
	# cv2.imshow('finalArray', finalArray)######################################

	color = cv2.cvtColor(finalArray, cv2.COLOR_GRAY2BGR)
	edges = cv2.Canny(finalArray,500,500)

	contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	###Find the bbox of contour return
	bbox = []
	for total in range(len(contours)):
		cnt = contours[total]
		x,y,w,h = cv2.boundingRect(cnt)
		bbox.append([x, y, x+w, y+h])
		cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),2)

	###Attempt to find the largest bbox/segmentation
	biggestArea = [0, 0, 0, 0]
	area = 0
	for x in range(len(bbox)):
		w = bbox[x][2] - bbox[x][0]
		h = bbox[x][3] - bbox[x][1]
		thisarea = w
		dist = math.sqrt(w**2 + h**2)

		if thisarea > area:
			area = thisarea
			biggestArea = bbox[x]

	print "area: ", area
	print "biggestArea: ", biggestArea
	cv2.rectangle(color,(biggestArea[0], biggestArea[1]),(biggestArea[2], biggestArea[3]),(0, 255, 255),2)

	if area == 0:	
		print "nothing matched. area was 0. black image"
		print "use edges and find contours"

		cv2.imshow('smolblur1', blur1) ###############################
		cv2.imshow('cutblur2', blur2)##################################
		cv2.imshow('difference', difference) #####################

		blurCopyAgain = difference
		for y in range(difference.shape[0]):
			for x in range(difference.shape[1]):
				diff = blur2[y][x] - blur1[y][x]
				if diff < 100 | diff > 80:
					blurCopyAgain[y][x] = 0
				else:
					blurCopyAgain[y][x] = diff
		cv2.imshow('blurCopy try again?', blurCopyAgain) ####################

		diffColor = cv2.cvtColor(blurCopyAgain, cv2.COLOR_GRAY2BGR)
		cv2.imshow('diffColor', diffColor)
		diffEdges = cv2.Canny(diffColor,100,200)
		cv2.imshow("diffEdges", diffEdges)

		diffContours, diffHierarchy = cv2.findContours(diffEdges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		print "diffContours: ", len(diffContours)

		npbbox = []
		for total in range(len(diffContours)):
			diffcnt = diffContours[total]
			x,y,w,h = cv2.boundingRect(diffcnt)
			npbbox.append([x, y, x+w, y+h])
			cv2.rectangle(diffColor,(x,y),(x+w,y+h),(0,255,0),2)

		npbbox = np.array(npbbox)
		new = npbbox[npbbox[:, 0].argsort()]


		##ALSO TEST WITH MORE CASES
		##ALSO TEST CROP WHOLE WIDE PANEL VS FRCNN SEGMENTATION.
		# largestArray = [diffColor.shape[1]-1, diffColor.shape[1], diffColor.shape[0], diffColor.shape[0]-1]
		largestArray = [diffColor.shape[1]-1, diffColor.shape[0]-1, diffColor.shape[1], diffColor.shape[0], ]
		print "LargestArray: ", largestArray
		for x in range(len(new)):
			add = find(new[x], largestArray)
			largestArray = add
			print "largestArray so far: ", largestArray

		print "largestArray: ", largestArray
		cv2.rectangle(diffColor,(largestArray[0], largestArray[1]),(largestArray[2], largestArray[3]),(0, 0, 255),2)
	
		cv2.imshow('diffColor', diffColor) #################

	print "----------------"

	# cv2.imshow('difference', difference) #####################
	cv2.imshow('color', color) ##########################

	cv2.waitKey(0)
	cv2.destroyAllWindows()








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