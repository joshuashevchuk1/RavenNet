import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

class LSTMModel(gluon.HybridBlock):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        with self.name_scope():
            self.lstm = nn.LSTM(hidden_size, num_layers=num_layers, dropout=dropout, layout='NTC')
            self.dense = nn.Dense(output_size)

    def hybrid_forward(self, F, x):
        x = self.lstm(x)
        x = self.dense(x[:, -1, :])  # Take the last time step
        return x

if __name__ == "__main__":
    # Parameters
    input_size = 1
    hidden_size = 50
    output_size = 1
    num_layers = 2

    # Instantiate the model
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    model.initialize()

    # Generate random training data
    x_train = mx.nd.random.uniform(shape=(100, 10, 1))  # (samples, timesteps, features)
    y_train = mx.nd.random.uniform(shape=(100, 1))      # (samples, output_dimension)

    # Define loss function and optimizer
    loss_fn = gluon.loss.L2Loss()
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})

    # Training loop
    for epoch in range(5):
        with autograd.record():
            output = model(x_train)
            loss = loss_fn(output, y_train)
        loss.backward()
        trainer.step(batch_size=100)
        print(f'Epoch [{epoch + 1}/5], Loss: {loss.mean().asscalar():.4f}')
