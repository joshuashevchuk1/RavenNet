from mxnet.gluon import nn, rnn, Block

#Define a simple LSTM model
class SimpleLSTM(Block):
    def __init__(self, hidden_size, num_layers, **kwargs):
        super(SimpleLSTM, self).__init__(**kwargs)
        self.lstm = rnn.LSTM(hidden_size, num_layers, layout='NTC')  # NTC layout: (batch_size, sequence_length, input_size)
        self.dense = nn.Dense(1)  # Output layer to map LSTM output to desired shape

    def forward(self, x):
        x = self.lstm(x)  # Pass input through LSTM
        return self.dense(x)  # Pass LSTM output through a dense layer