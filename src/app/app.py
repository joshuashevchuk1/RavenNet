from flask import Flask

class Ravenflask():
    def __init__(self, port):
        self.port = port
        self.app = Flask(__name__)

    def home(self):
        return "ok", 200

    def health_check(self):
        return "healthcheck", 200

    def add_routes(self):
        self.app.add_url_rule('/', 'home', self.home(), methods=["GET"])
        self.app.add_url_rule('/healthCheck', 'health_check', self.health_check, methods=["GET"])