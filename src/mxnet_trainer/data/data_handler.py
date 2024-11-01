import mxnet as mx
import os

class DataHandler:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        if os.path.exists(self.data_path):
            data = mx.nd.load(self.data_path)
            return data['x_train'], data['y_train']
        else:
            x_train = mx.nd.random.uniform(shape=(100, 10, 1))
            y_train = mx.nd.random.uniform(shape=(100, 1))
            self.save_data(x_train, y_train)
            return x_train, y_train

    def save_data(self, x_train, y_train):
        mx.nd.save(self.data_path, {'x_train': x_train, 'y_train': y_train})
