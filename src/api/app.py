from flask import Flask

import src.api.routes as routes

class Ravenflask():
    def __init__(self, port):
        self.port = port
        self.app = Flask(__name__)

    def add_routes(self):
        self.app.add_url_rule('/', 'home', routes.home, methods=["GET"])
        self.app.add_url_rule('/healthCheck', 'health_check', routes.health_check, methods=["GET"])

    def run_server(self):
        self.add_routes()
        self.app.run("0.0.0.0", self.port, debug=True)