import torch
import torch.nn as nn
import numpy as np

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        out = self.fc(output[:, -1, :])
        return out

def prepare_lstm_data(df, feature_cols, target_col, sequence_length=10):
    data = df[feature_cols + [target_col]].values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :-1])
        y.append(data[i+sequence_length, -1])
    X, y = np.array(X), np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def train_lstm_model(X_train, y_train, input_size, epochs=20, lr=0.001):
    model = LSTMRegressor(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f\"Epoch {epoch}, Loss: {loss.item():.4f}\")

    return model

def predict_lstm(model, X):
    model.eval()
    with torch.no_grad():
        preds = model(X).squeeze().numpy()
    return preds