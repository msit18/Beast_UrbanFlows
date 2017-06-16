import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def findMaxBBox(boxA, boxB):
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
		return findMaxBBox(boxA, boxB)
	else:
		print "There was no overlap"
		return 0

###Blurring and finding edges
def blurAndThresholdNoise(image1blur, image2blur, highThresh, lowThresh):
	print "highThresh: ", highThresh
	print "lowThresh: ", lowThresh
	npblur1 = np.asarray(image1blur)
	npblur2 = np.asarray(image2blur)
	npdiff = npblur2-npblur1
	cv2.imshow('npdiff', npdiff)

	_threshold = np.where(npdiff < highThresh, 0, npdiff)
	thresholdImage = np.where(_threshold < lowThresh, 0, _threshold)
	# cv2.imshow('_threshold', _threshold)######################################
	cv2.imshow('thresholdImage', thresholdImage)######################################

	return thresholdImage

def contoursToBBox(contours, image):
	bboxArray = []
	for c in range(len(contours)):
		cnt = contours[c]
		x,y,w,h = cv2.boundingRect(cnt)
		bboxArray.append([x, y, x+w, y+h])
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	return bboxArray

def findLargestXBbox(array, image):
	biggestAreaBbox = [0, 0, 0, 0]
	biggestArea = 130
	for x in range(len(array)):
		w = array[x][2] - array[x][0]

		if w > biggestArea:
			biggestArea = w
			biggestAreaBbox = array[x]

	print "area: ", biggestArea
	print "biggestArea: ", biggestAreaBbox
	cv2.rectangle(image,(biggestAreaBbox[0], biggestAreaBbox[1]),(biggestAreaBbox[2], biggestAreaBbox[3]),(0, 255, 255),2)
	return biggestAreaBbox

#include filter in this to make sure it returns a large enough bbox
def findBiggestBbox(array, image, maxXCoord, maxYCoord):
	defaultArray = [maxXCoord, maxYCoord, 0, 0]
	currentLargestBbox = defaultArray

	for x in range(len(array)):
		resultBbox = findMaxBBox(array[x], currentLargestBbox)
		print "largestBbox so far: ", resultBbox
		currentLargestBbox = resultBbox

	w = currentLargestBbox[2] - currentLargestBbox[0]

	print "This was the width value: ", w
	print "maxXCoord: ", maxXCoord
	print "maxXCoord*0.75: ", maxXCoord*0.75
	print "percentage of detection: ", float(w)/maxXCoord
	print "Compare width: ", maxXCoord*0.75 < w
	# print "TF: ", maxXCoord*0.75 < w
	if (maxXCoord*0.75 < w) == False:
		print "IS TRUE. BOX IS TOO SMALL. RETURNING FALSE"
		currentLargestBbox = defaultArray
		
	print "largestBbox: ", currentLargestBbox
	cv2.rectangle(image,(currentLargestBbox[0], currentLargestBbox[1]),(currentLargestBbox[2], currentLargestBbox[3]),(255, 0, 0),2)
	return currentLargestBbox

def recursiveFailSafe(blur1, blur2, smolSlice):
	print "RUNNING RECURSIVEFAILSAFE"
	highEdgeLimit = 100
	lowEdgeLimit = 80
	largestXBbox = [0,0,0,0]

	while (highEdgeLimit < 255) & (lowEdgeLimit >= 0) & (largestXBbox == [0,0,0,0]):
		cv2.destroyAllWindows()
		print "nothing matched. area was 0. black image"
		print "use edges and find contours"

		###Blurring
		diffImage = blur1
		for y in range(len(diffImage)):
			for x in range(len(diffImage[1])):
				diff = blur2[y][x] - blur1[y][x]
				if diff < highEdgeLimit | lowEdgeLimit > 80:
					diffImage[y][x] = 0
				else:
					diffImage[y][x] = diff
		cv2.imshow('diffImage try again?', diffImage) ####################

		#Edges and finding contours
		diffEdges = cv2.Canny(diffImage, 100, 200)
		cv2.imshow('diffEdges', diffEdges)
		diffContours, diffHierarchy = cv2.findContours(diffEdges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		#Find bbox of all contour returns, and draw all bbox found
		diffColor = cv2.cvtColor(diffImage, cv2.COLOR_GRAY2BGR)
		# npbbox = []
		npbbox = contoursToBBox(diffContours, diffColor)
		print "NPBBOX : ", npbbox

		runAgain = False
		if len(npbbox) > 0:
			npbbox = np.array(npbbox)
			sortedNpBbox = npbbox[npbbox[:, 0].argsort()]

			print "DiffColor1: ", diffColor.shape[1]
			print "diffColor2: ", diffColor.shape[0]

			largestBboxTotal = findBiggestBbox(sortedNpBbox, diffColor, diffColor.shape[1], diffColor.shape[0])
			print "largestBboxTotal: ", largestBboxTotal
			print "TF: ", largestBboxTotal != [diffColor.shape[1], diffColor.shape[0],0,0]

			if largestBboxTotal != [diffColor.shape[1], diffColor.shape[0],0,0]:
				cv2.imshow('diffColor', diffColor) #################
				return largestBboxTotal
			else:
				runAgain = True
		else:
			runAgain = True

		if runAgain == True:
			print "BBOX IS EMPTY. THRESHOLD NEEDS TO BE FURTHER AJUSTED"
			##adjust the parameters for filtering
			highEdgeLimit += 20
			lowEdgeLimit -= 20
			print "HighEdgeLimit: ", highEdgeLimit
			print "lowEdgeLimit: ", lowEdgeLimit

			if (highEdgeLimit > 225) & (lowEdgeLimit < 0):
				print "RECURSIVE HAS MAXED OUT PARAMETERS. CANNOT DETECT EDGES"
				break
			else:
				if highEdgeLimit > 255:
					print "highEdgeLimit is too high. Setting"
					highEdgeLimit = 250
				elif lowEdgeLimit < 0:
					print "lowEdgeLimit is too low. Setting"
					lowEdgeLimit = 10

				print "highEdgeLimit has increased: ", highEdgeLimit
				print "lowEdgeLimit has changed: ", lowEdgeLimit
	


def main():
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
	# analyzeImage = cv2.imread('../04-26_15.00.00_splice/000000.jpg')
	# analyzeImage = cv2.imread('../04-26_15.00.00_splice/000450.jpg')
	#16:00
	# analyzeImage = cv2.imread('../04-26_16.00.00_splice/000000.jpg')
	# analyzeImage = cv2.imread('../04-26_16.00.00_splice/000900.jpg')
	# analyzeImage = cv2.imread('../04-26_16.00.00_splice/001125.jpg')
	#17:00
	analyzeImage = cv2.imread('../04-26_17.00.00_splice/001350.jpg')

	## 13:00
	# array = [[282, 414, 621, 551], [649, 400, 1016, 537], [1037, 375, 1350, 546]]
	## 14:00
	# array = [[ 1019.9876709 ,   380.28192139,  1356.57019043,   550.44891357],
			 # [  619.75952148,   399.24371338,  1064.24182129,   540.62957764]]
	## 15:00
	# array = [[ 1045.41955566,   355.86233521,  1358.03820801,   545.76141357],
			 # [ 1335.08007812,   438.89526367,  1480.54150391,   523.03808594],
			 # [ 370.89178467,  407.24188232,  563.11383057,  564.81378174]]
	# array = [[ 1062.32592773,   370.96105957,  1339.44604492,   568.22747803],
			 # [ 1341.07568359,   451.25686646,  1491.05078125,   539.85400391]]
	##16:00
	# array = [[  673.87365723,   409.62060547,  1033.48718262,   553.62713623],
			# [ 298.14996338,  424.37869263,  612.6762085 ,  584.33312988],
			# [ 1030.12219238,   365.12265015,  1354.13317871,   566.68762207]]
	# array = [[ 678.86743164,  405.30609131,  995.79394531,  547.12176514],
			# [ 256.40667725,  418.96252441,  613.74786377,  560.73895264], 
			# [ 1219.65112305,   391.85281372,  1434.32226562,   548.12750244]]
	# array = [[ 673.22119141,  406.94473267,  998.57727051,  546.52600098],
			 # [ 284.42590332,  417.16751099,  590.66571045,  551.29376221]]
	##17:00
	array = [[ 281.34237671,  417.8475647 ,  617.19787598,  551.34368896],
			 [  44.0845871 ,  411.99499512,  226.28234863,  549.74505615]]

	# x = 0
	# if x == 0:
	for x in range(len(array)):
		###Image slicing and blurring
		smolSlice = analyzeImage[array[x][1]:array[x][3], array[x][0]:array[x][2]]
		gray1 = cv2.cvtColor(smolSlice,cv2.COLOR_BGR2GRAY)
		blur1 = cv2.blur(gray1,(20,20))
		blur1_copy = blur1
		cv2.imshow('smolSlice', smolSlice)
		# cv2.imshow('smolblur1', blur1) ###############################

		cutSlice = merge[array[x][1]:array[x][3], array[x][0]:array[x][2]]
		gray2 = cv2.cvtColor(cutSlice,cv2.COLOR_BGR2GRAY)
		blur2 = cv2.blur(gray2,(20,20))
		blur2_copy = blur2
		# cv2.imshow('cutSlice', cutSlice)
		# cv2.imshow('cutblur2', blur2)##################################		

		differenceImage = blurAndThresholdNoise(blur1_copy, blur2_copy, 240, 10)
		# cv2.imshow("differenceImage", differenceImage)

		#Edges and finding contours
		edges = cv2.Canny(differenceImage,500,500)
		contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		#Find bbox of all contour returns, and draw all bbox found
		color = cv2.cvtColor(differenceImage, cv2.COLOR_GRAY2BGR)
		bbox = contoursToBBox(contours, color)

		#Find widest bbox and draw in yellow on image
		largestXBbox = findLargestXBbox(bbox, color)
		cv2.imshow('color', color) ##########################

		cv2.waitKey(0)

		if largestXBbox == [0,0,0,0]:
			recursiveFailSafe(blur1, blur2, smolSlice)


			# cv2.destroyAllWindows()
			# print "nothing matched. area was 0. black image"
			# print "use edges and find contours"

			# # cv2.imshow('smolblur1', blur1) ###############################
			# # cv2.imshow('cutblur2', blur2)##################################
			# # cv2.imshow('difference', difference) #####################

			# # difference = blur2_copy-blur1_copy #used as a baseline here.
			# # cv2.imshow('difference', difference)

			# # diffImage = blurAndThresholdNoise(blur1_copy, blur2_copy, 140, 60)
			# # cv2.imshow('diffImage', diffImage)

			# diffImage = blur1_copy
			# for y in range(len(diffImage)):
			# 	for x in range(len(diffImage[1])):
			# 		diff = blur2[y][x] - blur1[y][x]
			# 		if diff < 100 | diff > 80:
			# 			diffImage[y][x] = 0
			# 		else:
			# 			diffImage[y][x] = diff
			# cv2.imshow('diffImage try again?', diffImage) ####################

			# #Edges and finding contours
			# diffEdges = cv2.Canny(diffImage, 100, 200)
			# diffContours, diffHierarchy = cv2.findContours(diffEdges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

			# #Find bbox of all contour returns, and draw all bbox found
			# diffColor = cv2.cvtColor(diffImage, cv2.COLOR_GRAY2BGR)
			# npbbox = contoursToBBox(diffContours, diffColor)
			# print "NPBBOX : ", npbbox

			# if len(npbbox) > 0:
			# 	npbbox = np.array(npbbox)
			# 	sortedNpBbox = npbbox[npbbox[:, 0].argsort()]

			# 	print "DiffColor1: ", diffColor.shape[1]
			# 	print "diffColor2: ", diffColor.shape[0]

			# 	largestBboxTotal = findMaxBbox(sortedNpBbox, diffColor, diffColor.shape[1], diffColor.shape[0])
			# else:
			# 	print "BBOX IS EMPTY. THRESHOLD NEEDS TO BE FURTHER AJUSTED"

			# cv2.imshow('diffColor', diffColor) #################
		cv2.imshow('smolSlice', smolSlice)
		print "NEXT IMAGE------------------------"


		cv2.waitKey(0)
		cv2.destroyAllWindows()

main()