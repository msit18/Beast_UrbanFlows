from twisted.internet.protocol import Protocol

class Echo(Protocol):

	def dataReceived (self, data):
		self.transport.write(data)
