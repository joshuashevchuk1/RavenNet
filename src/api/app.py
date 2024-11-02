from flask import Flask

import routes

class Ravenflask():
    def __init__(self, port):
        self.port = port
        self.app = Flask(__name__)

    def add_routes(self):
        self.app.add_url_rule('/', 'home', routes.home(), methods=["GET"])
        self.app.add_url_rule('/healthCheck', 'health_check', routes.health_check(), methods=["GET"])