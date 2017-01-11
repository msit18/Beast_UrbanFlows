#Still being written: callback server.  Used with testUploadSpeed2.py to send data and messages

#TODO: Not implemented: A scaleable method of keeping track of the clusters. Nahom proposed
#having a team leader pi who is responsible for pinging its teammates when the server detects
#that a connection has been lost. Instead of pinging all of the clients, just ping 1/4 (the
#team leaders) who then ping their team members and report back if they have lost someone.

#Written by Michelle Sit
#Many thanks to Vlatko Klabucar for helping me with the HTTP part!  Also many thanks to Nahom Marie
#for helping me with the architecture of this system!

from twisted.internet.protocol import Factory
from twisted.internet import reactor, protocol, defer, threads
import sys, time, os
from masterVariables2 import MasterVariables

from twisted.web.server import Site
from twisted.web.resource import Resource

import cgi
import subprocess
import datetime

#sets up the Protocol class
class DataFactory(Factory):
	numConnections = 0

	def __init__(self, data=None):
		self.data = data
		self.ipDictionary = {}
		self.checkCamPi = 0
		self.finished = 0
		# self.timesToTakeVideo = ["06/24/16 17:58:00", "06/24/16 18:58:00", "06/24/16 19:58:00", \
		# 						"06/24/16 20:58:00", "06/24/16 21:58:00", "06/24/16 22:58:00", "06/25/16 05:58:00", "06/25/16 06:58:00", \
		# 						"06/25/16 07:58:00", "06/25/16 08:58:00", "06/25/16 09:58:00", "06/25/16 10:58:00", "06/25/16 11:58:00", \
		# 						"06/25/16 12:58:00", "06/25/16 13:58:00", "06/25/16 14:58:00", "06/25/16 15:58:00", "06/25/16 16:58:00", \
		# 						"06/25/16 17:58:00", "06/25/16 18:58:00", "06/25/16 19:58:00", "06/25/16 20:58:00", "06/25/16 21:58:00", \
		# 						"06/25/16 22:58:00", "06/26/16 05:58:00", "06/26/16 06:58:00", "06/26/16 07:58:00", "06/26/16 08:58:00", \
		# 						"06/26/16 09:58:00", "06/26/16 10:58:00", "06/26/16 11:58:00", "06/26/16 12:58:00", "06/26/16 13:58:00", \
		# 						"06/26/16 14:58:00", "06/26/16 15:58:00", "06/26/16 16:58:00", "06/26/16 17:58:00", "06/26/16 18:58:00", \
		# 						"06/26/16 19:58:00", "06/26/16 20:58:00", "06/26/16 21:58:00", "06/26/16 22:58:00", "06/27/16 06:58:00", \
		# 						"06/26/16 07:58:00", "06/26/16 08:58:00", "06/26/16 09:58:00", "06/26/16 10:58:00", "06/26/16 11:58:00"]
		# self.videoTotalTimeSecDuration = [1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020, \
		# 								 1020, 1020, 1020, 1020]
		self.timesToTakeVideo = ["01/11/17 15:21:00", "01/11/17 15:21:10"]

		self.videoTotalTimeSecDuration = [300, 300]


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
		if msgFromClient[0] == "ip":
			print "FOUND AN IP"
			self.name = msgFromClient[2]
			self.factory.ipDictionary[self.name] = self
			print "Echoers: ", self.factory.ipDictionary
			print "RUNNING CHECKCONNECTIONS"
			if len(self.factory.ipDictionary) > (totalNumRaspies-1): #Set value to total number of Raspies -1
				print "verifying Connections with connections"
				# self.verifyConnections()
				self.startProgram()

		elif msgFromClient[0] == 'CAMERROR':
			print "CAM ERROR FROM {1} PICAMERA at {0}".format(time.strftime("%Y-%m-%d-%H:%M:%S"), msgFromClient[1])

		elif msgFromClient[0] == "UPLOADERROR":
			print "UPLOAD ERROR FROM {1} PICAMERA at {0}".format(time.strftime("%Y-%m-%d-%H:%M:%S"), msgFromClient[1])

		elif msgFromClient[0] == 'checkCamPi':
			self.factory.checkCamPi += 1
			if self.factory.checkCamPi > (totalNumRaspies-1):
				print "All raspies are ready to start process"
				print "Running send cmds"
				self.startProgram()
			else:
				print "Still waiting on other raspies to connect. {0} raspies are ready".format(self.factory.checkCamPi)

		elif msgFromClient[0] == 'checkingUploadedFileSize':
			serverFileSize = self.checkFileSize(msgFromClient[1])
			print "sending message: filesize is {0}".format(str(serverFileSize))
			self.transport.write(str(serverFileSize))

		elif msgFromClient[0] == 'finished':
			self.factory.finished += 1
			if self.factory.finished > (totalNumRaspies-1):
				print "All raspies are finished--------------------------"
				if len(self.factory.timesToTakeVideo) > 0:
					self.factory.finished = int(f.numRaspiesInCluster)
					self.factory.checkCamPi = 0
					f.ServerStartTime = self.factory.timesToTakeVideo.pop(0)
					newStartTimeConvert = datetime.datetime.strptime(f.ServerStartTime, "%x %X")
					f.ServerTotalTimeSec = self.factory.videoTotalTimeSecDuration.pop(0)
					print "Running next video time: " + str(f.ServerStartTime) + ". " +\
						str(len(self.factory.timesToTakeVideo)) + " remaining runs left."
					while datetime.datetime.today() < newStartTimeConvert:
						pass
					else:
						# self.verifyConnections()
						self.startProgram()
				else:
					print "There are no more times left to take. System has finished."
			else:
				print "Still waiting on other raspies to finish taking or uploading pictures. {0} raspies are finished".format(self.factory.finished)

		else:
			print "Time: {0}. I don't know what this is: {1}".format(time.strftime("%Y-%m-%d-%H:%M:%S"), data)

	#USE THIS FOR THE LARGER SCALEABLE SYSTEM
	# def updatingIPDictionary(self, dictionary, piGroup, ipAddr):
	# 	print "RUNNING GOTIP"
	# 	#adds key and a list containing IP address
	# 	if (piGroup in dictionary) == False:
	# 		print "I didn't have this cluster key for {0}".format(dictionary)
	# 		dictionary[piGroup] = [ipAddr]
	# 	#appends new IP to the end of the key's list
	# 	elif (piGroup in dictionary) == True:
	# 		print "it's true! I have this cluster in my keys for {0}".format(dictionary)
	# 		dictionary[piGroup].append(ipAddr)
	# 	else:
	# 		print "Got something that wasn't an IP. Adding to dict anyway for {0}".format(dictionary)
	# 		dictionary[piGroup] = [ipAddr]
	# 	print dictionary

	def verifyConnections(self):
		for echoer in self.factory.ipDictionary:
			sendMsg = "checkCamera " + time.strftime("%x %X")
			print "sendMsg for verify ", sendMsg
			self.factory.ipDictionary[echoer].transport.write(sendMsg)

	def startProgram(self):
		print "STARTTAKINGPICTURES"
		for echoer in self.factory.ipDictionary:
			sendMsg = "startProgram {0}".format(f.getParam())
			print sendMsg
			self.factory.ipDictionary[echoer].transport.write(sendMsg)

	def checkFileSize(self, filename):
		try:
			fileSize = os.path.getsize("{1}{0}".format(filename, serverSaveFilePath))
			return "checkFileSizeIsCorrect {0} {1}".format(filename, fileSize)
		except:
			print "Could not upload to server"
			return "uploadingError"

#Used for HTTP network.  Receives images and saves them to the server
class UploadImage(Resource):

	def render_GET(self, request):
		print "RENDER GETTING"
		return '<html><body><p>This is the server for the MIT SENSEable City Urban Flows Project.'\
		'  It receives images and saves them to the server.</p></body></html>'

	def render_POST(self, request):
		name = request.getHeader('filename')
		print "RENDER Posting: {0}".format(name)
		with open(name, "wb") as file:
			file.write(request.content.read())
		print "finished writing file"
		return '<html><body>Image uploaded :) </body></html>'

if __name__ == '__main__':
	#log = open('ServerLog-{0}.txt'.format(time.strftime("%Y-%m-%d-%H:%M:%S")), 'w')
	f = MasterVariables()
	f.userInput()

	ip_address = subprocess.check_output("hostname --all-ip-addresses", shell=True).strip()
	serverIP = ip_address.split()[0]
	totalNumRaspies = int(f.numRaspiesInCluster)
	serverSaveFilePath = "/home/msit/"

	#TCP network
	d = defer.Deferred()
	b = DataFactory()
	reactor.listenTCP(8888, b, 200, serverIP)

	#HTTP network
	a = UploadImage()
	root = Resource()
	root.putChild("upload-image", a)
	factory = Site(root)
	reactor.listenTCP(8880, factory, 200, serverIP)

	print "SERVER IP IS: ", serverIP
	print "SERVER SAVEFILEPATH IS: ", serverSaveFilePath

	reactor.run()
