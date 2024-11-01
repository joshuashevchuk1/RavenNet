from mxnet import autograd, gluon

# this class takes of training of uploaded data
class LSTMTrainer:
    def __init__(self, model, data_handler, config):
        self.model = model
        self.data_handler = data_handler
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.loss_fn = gluon.loss.L2Loss()
        self.trainer = gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': config['learning_rate']})

    def train(self):
        x_train, y_train = self.data_handler.load_data()
        for epoch in range(self.epochs):
            with autograd.record():
                output = self.model(x_train)
                loss = self.loss_fn(output, y_train)
            loss.backward()
            self.trainer.step(batch_size=self.batch_size)
            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.mean().asscalar():.4f}')
