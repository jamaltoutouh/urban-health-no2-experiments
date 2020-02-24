import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

#cudnn.benchmark = True

data_txt = '../data/txt/'
data_csv = '../data/csv/'

dataset = pd.read_csv(data_csv+'sequence_air_all.csv', header=0, index_col=0)

dataset.replace(-1, dataset.mean())

all_data = dataset['measure'].values.astype(float)

print(len(all_data))


cortar_data = all_data[-5000:]
all_data = cortar_data

test_data_size = int(0.1 * len(all_data))


train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

print(len(train_data))
print(len(test_data))

# Normalize data [-1, 1]
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# Dataset to Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# Set train window
train_window = 24

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

print('Defining the RNN')

# # Create LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(device),
                            torch.zeros(1,1,self.hidden_layer_size).to(device))

        self.lstm.to(device)
        self.linear.to(device)


    def forward(self, input_seq):
        input_seq.to(device)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(input_seq.size(0),1, -1) , self.hidden_cell)
        lstm_out = lstm_out.to(device)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    # def forward2(self, input, future=0, y=None):
    #     input.to(device)
    #     outputs = []        # reset the state of LSTM
    #     # the state is kept till the end of the sequence
    #     h_t = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.float32).to(device)
    #     c_t = torch.zeros(input.size(0), self.hidden_layer_size, dtype=torch.float32).to(device)
    #
    #     for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
    #         h_t, c_t = self.lstm(input_t, (h_t, c_t))
    #         output = self.linear(h_t)
    #         outputs += [output]
    #
    #     for i in range(future):
    #         if y is not None and random.random() > 0.5:
    #             output = y[:, [i]]  # teacher forcing
    #         h_t, c_t = self.lstm(output, (h_t, c_t))
    #         output = self.linear(h_t)
    #         outputs += [output]
    #     outputs = torch.stack(outputs, 1).squeeze(2)
    #     return outputs

print('Model defined')
model = LSTM()
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 150


print('Start training')
for i in range(epochs):
    j = 0
    for seq, labels in train_inout_seq:
        seq.to(device)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        seq = seq.to(device)
        labels = labels.to(device)
        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

        # if j%10 == 0:
        #     print(j)
        # j += 1

    if i%1 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')