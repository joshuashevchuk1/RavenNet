import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

if __name__ == "__main__":
    # Parameters
    input_size = 1
    hidden_size = 50
    output_size = 1
    num_layers = 2

    # Instantiate the model
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Generate random training data
    x_train = torch.rand(100, 10, 1)  # (samples, timesteps, features)
    y_train = torch.rand(100, 1)      # (samples, output_dimension)

    # Training loop
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/5], Loss: {loss.item():.4f}')
