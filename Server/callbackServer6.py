#Used with testUploadSpeed4

#TODO: Not implemented: A scaleable method of keeping track of the clusters. Nahom proposed
#having a team leader pi who is responsible for pinging its teammates when the server detects
#that a connection has been lost. Instead of pinging all of the clients, just ping 1/4 (the
#team leaders) who then ping their team members and report back if they have lost someone.

#Written by Michelle Sit

from twisted.internet.protocol import Factory
from twisted.internet import reactor, protocol, defer, threads
from twisted.web.server import Site
from twisted.web.resource import Resource

import sys, subprocess, os
import datetime, time

from masterVariables2 import MasterVariables

#sets up the Protocol class
class DataFactory(Factory):
	numConnections = 0

	def __init__(self, data=None):
		self.data = data
		self.ipDictionary = {}
		self.checkCamPi = 0 #running tally of raspies for current process (waiting for initial connections or received all times)
		self.finished = 0

		# self.timesToTakeVideo = "02/3/17 12:30:00 02/3/17 12:45:00 02/3/17 01:00:00 "\
		# 						"02/3/17 01:15:00 02/3/17 01:30:00 02/3/17 01:45:00 "
		self.timesToTakeVideo = ""
		# self.videoTotalTimeSecDuration = 10

	def buildProtocol(self, addr):
		return DataProtocol(self, d)

class DataProtocol (protocol.Protocol):

	def __init__(self, factory, d):
		self.factory = factory
		self.d = defer.Deferred()
		self.name = None

	def connectionMade(self):
		self.factory.numConnections += 1
		print "Connection made at {0}. Number of active connections: {1}".format(time.strftime("%Y-%m-%d-%H:%M:%S"), self.factory.numConnections)

	def connectionLost(self, reason):
		self.factory.numConnections -= 1
		if self.name in self.factory.ipDictionary:
			del self.factory.ipDictionary[self.name]
		print "Echoers: ", self.factory.ipDictionary
		print "Connection lost at {0}. Number of active connections: {1}".format(time.strftime("%Y-%m-%d-%H:%M:%S"), self.factory.numConnections)

	def dataReceived(self, data):
		print "DATARECEIVED. Server received data: {0}".format(data)
		msgFromClient = [data for data in data.split()]
		if msgFromClient[0] == "clientName":
			print "FOUND AN IP"
			self.name = msgFromClient[2]
			self.factory.ipDictionary[self.name] = self
			print "Echoers: ", self.factory.ipDictionary
			print "RUNNING CHECKCONNECTIONS"
			if len(self.factory.ipDictionary) > (totalNumRaspies-1): #Set value to total number of Raspies -1
				print "All raspies are ready to start process"
				print "Starting program."
				print "Step 1/2: Sending all timesToTakeVideo:"
				self.factory.checkCamPi = 0
				for echoer in self.factory.ipDictionary:
					sendRecordTimes = "recordTimes {0}".format(self.factory.timesToTakeVideo)
					print sendRecordTimes
					self.factory.ipDictionary[echoer].transport.write(sendRecordTimes)
			else:
				print "Step 1/2 waiting on other raspies to connect. {0} raspies are ready".format(self.factory.checkCamPi)

		elif msgFromClient[0] == 'receivedAllTimesReadytoStart':
			self.factory.checkCamPi += 1
			if self.factory.checkCamPi > (totalNumRaspies-1):
				print "All raspies have received record times."
				print "Step 2/2: Starting recording"
				self.startProgram()
			else:
				print "Step 2/2 waiting on other raspies to receive times. {0}/{1} raspies are ready".format(self.factory.checkCamPi, totalNumRaspies)


		elif msgFromClient[0] == 'finished':
			if len(msgFromClient) > 0:
				print "other msgs: ", msgFromClient[1:]
			self.factory.finished += 1
			if self.factory.finished > (totalNumRaspies-1):
				print "All raspies are finished--------------------------"
				if len(self.factory.timesToTakeVideo) > 0:
					print "PROCESS SENT FINISHED."
					# self.factory.finished = int(f.numRaspiesInCluster)
					# self.factory.checkCamPi = 0
					# f.ServerStartTime = self.factory.timesToTakeVideo.pop(0)
					# newStartTimeConvert = datetime.datetime.strptime(f.ServerStartTime, "%x %X")
					# f.ServerTotalTimeSec = self.factory.videoTotalTimeSecDuration.pop(0)
					# print "Running next video time: " + str(f.ServerStartTime) + ". " +\
					# 	str(len(self.factory.timesToTakeVideo)) + " remaining runs left."
				else:
					print "There are no more times left to take. System has finished."
			else:
				print "Still waiting on other raspies to finish taking or uploading pictures. {0} raspies are finished".format(self.factory.finished)


		#Error messages
		elif msgFromClient[0] == 'CAMERROR':
			print "CAM ERROR FROM {1} PICAMERA at {0}".format(time.strftime("%Y-%m-%d-%H:%M:%S"), msgFromClient[1])

		else:
			print "Time: {0}. I don't know what this is: {1}".format(time.strftime("%Y-%m-%d-%H:%M:%S"), data)

	def startProgram(self):
		print "STARTTAKINGPICTURES"
		for echoer in self.factory.ipDictionary:
			sendMsg = "startProgram {0}".format(f.getParam())
			print sendMsg
			self.factory.ipDictionary[echoer].transport.write(sendMsg)

if __name__ == '__main__':
	f = MasterVariables()
	f.userInput()

	ip_address = subprocess.check_output("hostname --all-ip-addresses", shell=True).strip()
	serverIP = ip_address.split()[0]
	totalNumRaspies = int(f.numRaspiesInCluster)
	serverSaveFilePath = "/media/msit/PhilipsData/TrafficIntersection17/" #Leave out dashes. Add dashes for client.
	#serverSaveFilePath = "/media/senseable-beast/beast-brain-1/Data/OneWeekData/tmp/"

	#TCP network
	d = defer.Deferred()
	b = DataFactory()
	reactor.listenTCP(8888, b, 200, serverIP)

	print "SERVER IP IS: ", serverIP
	print "SERVER SAVEFILEPATH IS: ", serverSaveFilePath

	reactor.run()
