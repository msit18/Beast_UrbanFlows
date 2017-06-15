import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('/home/msit/Desktop/04-26_14.00.00_splice/005625.jpg')
crop = img[300:600, 0:1600]

img2 = cv2.imread('/home/msit/Desktop/04-26_17.00.00_splice/000225.jpg')
crop2 = img2[300:600, 0:1600]

spot = img2[300:600, 950:1400]

merge = img.copy()
merge[300:600, 950:1400] = img2[300:600, 950:1400]

analyzeImage = cv2.imread('001350.jpg')

# array = [[ 247.84080505,  434.17144775,  626.03320312,  554.61505127],
# 		[  622.02294922,   405.94180298,  1028.23376465,   539.5322876 ],
# 		[ 1035.34484863,   379.25918579,  1355.61804199,   544.60784912]]

array = [[282, 414, 621, 551], [649, 400, 1016, 537], [1037, 375, 1350, 546]]

for x in range(len(array)):
	cv2.rectangle(merge, (int(array[x][0]), int(array[x][1])), (int(array[x][2]), int(array[x][3])), (255, 255, 255), 3)
	smolSlice = analyzeImage[array[0][1]:array[0][3], array[0][0]:array[0][2]]
	gray1 = cv2.cvtColor(smolSlice,cv2.COLOR_BGR2GRAY)
	blur1 = cv2.blur(gray1,(20,20))
	blur3 = cv2.blur(gray1,(5,5))
	# cv2.imshow('smolSlice', smolSlice)

	cutSlice = merge[array[0][1]:array[0][3], array[0][0]:array[0][2]]
	gray2 = cv2.cvtColor(cutSlice,cv2.COLOR_BGR2GRAY)
	blur2 = cv2.blur(gray2,(20,20))
	blur4 = cv2.blur(gray2,(5,5))
	# cv2.imshow('cutSlice', cutSlice)

	# difference = gray2-gray1
	difference = blur2-blur1
	diff_copy = difference.copy()

#######################################
	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()

	params.filterByColor = True
	params.blobColor = 255

	# Change thresholds
	params.minThreshold = 100
	params.maxThreshold = 10000000

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 150

	# # Filter by Circularity
	# params.filterByCircularity = True
	# params.minCircularity = 0.1

	# # Filter by Convexity
	# params.filterByConvexity = True
	# params.minConvexity = 0.87

	# # Filter by Inertia
	# params.filterByInertia = True
	# params.minInertiaRatio = 0.01

	# Create a detector with the parameters
	detector = cv2.SimpleBlobDetector(params)


	# Detect blobs.
	keypoints = detector.detect(difference)
	print "Keypts: ", keypoints

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
	# the size of the circle corresponds to the size of blob
	color = cv2.cvtColor(diff_copy, cv2.COLOR_GRAY2BGR)
	im_with_keypoints = cv2.drawKeypoints(color, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# # Show blobs
	# cv2.imshow("Keypoints", im_with_keypoints)
	# cv2.waitKey(0)






	# contours, hierarchy = cv2.findContours(diff_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# print "Contours: ", contours
	# print "Hiearchy: ", hierarchy

	# # diff_copy = difference.copy()
	# # cv2.drawContours(difference, contours, 0, (0,255,0), 6)

	# cnt = contours[4]
	# color = cv2.cvtColor(diff_copy, cv2.COLOR_GRAY2BGR)
	# x,y,w,h = cv2.boundingRect(cnt)
	# cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),2)

	
	# cv2.rectangle(color,(0,0),(100,100),(0,255,0),2)

	# # cv2.drawContours(diff_copy, [cnt], 0, (0,255,0), 3)

	cv2.imshow('difference', difference)
	difference2 = blur4-blur3
	# cv2.imshow('difference2', difference2)
	cv2.imshow('diff_copy', im_with_keypoints)

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
