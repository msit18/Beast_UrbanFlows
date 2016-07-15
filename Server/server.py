from twisted.web.server import Site
from twisted.web.resource import Resource
from twisted.internet import reactor

import cgi

class UploadImage(Resource):
    def render_GET(self, request):
        return '<html><body><p>To upload an image just send me a POST request.</p><p> P.S. don\'t forget the image :)</p></body></html>'

    def render_POST(self, request):
        file = open("uploaded-image.jpg","wb")
        file.write(request.content.read())
        return '<html><body>Image uploaded :) </body></html>'

root = Resource()
root.putChild("upload-image", UploadImage())
factory = Site(root)
reactor.listenTCP(8880, factory)
reactor.run()
