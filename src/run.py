#!/usr/bin/python

import multiprocessing

def run_flask_app():
    # server = pythonTrainerServer.Mapflask(config.mapflask_port)
    # mapflask_server.run_server()
    return

if __name__ == '__main__':
    flask_process = multiprocessing.Process(target=run_flask_app)
    # Start the Flask process
    flask_process.start()