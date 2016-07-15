#Written by Michelle Sit
#Stores the Finished variable for callbackServer4.py
#Also stores all the user inputs for the camera

#Nice TODO: Sanitize user input to make sure it is valid
#TODO: The inputted fps is not actually what the camera performs at

import time
import datetime

class MasterVariables():
	def __init__(self):
		self.ServerTotalTimeSec = ""
		self.ServerResW = ""
		self.ServerResH = ""
		self.ServerNumPics = ""
		self.ServerTimeInterval = ""
		self.ServerFrameRate = ""
		self.goInput = ""
		self.param = ""
		self.camVid = ""
		self.ServerStartTime = ""
		self.numRaspiesInCluster = ""

	def checkIfTimeIsValid(self):
		while True:
			self.ServerStartTime = raw_input("Enter date and time that you would like"\
				"to start the program (ex - 11/04/16 15:45:58). Current time is {0} : "\
				.format(time.strftime("%x %X")))
			try:
				userInputTime = datetime.datetime.strptime(self.ServerStartTime, "%x %X")
				if (userInputTime > datetime.datetime.today()) == True:
					break
			except ValueError:
				print "That is not a valid date or time. Please try again."

	def getParam(self):
		if self.camVid == "camera":
			self.param = "camera {0} {1} {2} {3} {4} {5} {6}".format(self.ServerTotalTimeSec,\
			 self.ServerResW, self.ServerResH, self.ServerNumPics, self.ServerTimeInterval,\
			 self.ServerFrameRate, self.ServerStartTime)
		elif self.camVid == "video":
			self.param = "video {0} {1} {2} {3} {4}".format(\
			self.ServerResW, self.ServerResH, self.ServerTotalTimeSec, self.ServerFrameRate, \
			self.ServerStartTime)
		return self.param

	def userInput(self):
		self.numRaspiesInCluster = raw_input('Enter how many raspies are in this cluster: ')
		self.camVid = raw_input ('Enter camera or video: ')
		self.checkIfTimeIsValid()
		#self.ServerStartTime = time.strftime("%x %X")
		print "Starting time: ", self.ServerStartTime
		if self.camVid == "camera":
			self.ServerTotalTimeSec = raw_input('Enter total run time in seconds: ')
			self.ServerResW = raw_input('Enter resolution width: ')
			self.ServerResH = raw_input('Enter resolution height: ')
			self.ServerNumPics = raw_input ('Enter number of pictures to take (fps): ')
			self.ServerTimeInterval = raw_input ("Enter time interval (seconds) for frames"\
			" to be taken in (fps): ")
			self.ServerFrameRate = raw_input ('Enter framerate: ')
			print "Thank you for your input. Please check the following"
			print "{0} | TotalTime(sec): {1} | ResW: {2} | ResH: {3} | NumPics: {4} | "\
			"TimeInterval(sec): {5} | FR: {6} | StartTime: {7} | NumRaspies: {8}".format(self.camVid, \
			self.ServerTotalTimeSec, self.ServerResW, self.ServerResH,\
			self.ServerNumPics, self.ServerTimeInterval, self.ServerFrameRate, self.ServerStartTime, \
			self.numRaspiesInCluster)

		elif self.camVid == "video":
			#self.ServerVidTimeSec = raw_input('Enter individual video time(sec): ')
			self.ServerResW = raw_input('Enter resolution width: ')
			self.ServerResH = raw_input('Enter resolution height: ')
			self.ServerTotalTimeSec = input('Enter total run time in seconds: ')
			self.ServerFrameRate = raw_input ('Enter framerate: ')
			print "Thank you for your input. Please check the following"
			print "{0} | ResW: {1} | ResH: {2} | TotalTime (sec): {3} |"\
			" FR: {4} | StartTime: {5} | NumRaspies: {6}"\
			.format(self.camVid, self.ServerResW,\
			self.ServerResH, self.ServerTotalTimeSec, self.ServerFrameRate, \
			self.ServerStartTime, self.numRaspiesInCluster)

		elif self.camVid == "multiplexer":
			print "still working on this feature. Please try again"
			self.userInput()

		else:
			print "Wrong input. Please try again"
			self.userInput()

		goInput = raw_input ('Run server? Yes or no: ')
		if goInput == "yes":
			print self.getParam()
			print "Running server now"
		elif goInput == "no":
			self.userInput()