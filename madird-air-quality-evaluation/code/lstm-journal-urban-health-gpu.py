import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

#training_set = pd.read_csv('airline-passengers.csv')
#training_set = pd.read_csv('shampoo.csv')

#training_set = training_set.iloc[:,1:2].values
#print(training_set)

import sys
import os
hidden_size = int(sys.argv[1])
look_back = int(sys.argv[2])
output_folder = sys.argv[3]
output_path = output_folder + '-' + str(hidden_size) + '-' + str(look_back)
os.mkdir(output_path)

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


def to_dataframe(actual, predicted):
    return pd.DataFrame({"actual": actual, "predicted": predicted})

data_csv = '../data/csv/'
# results_folder = '../data/results/'
dataset = pd.read_csv(data_csv+'035-08-sequence_air_all.csv', header=0, index_col=0)
dataset.replace(-1, dataset.mean())


whole_set = dataset['measure'].values
print(whole_set.shape)


training_set = dataset[dataset.index < "2017-12-01"]['measure'].values
print(training_set.shape)

testing_set = dataset[(dataset.index >= "2017-12-01") & (dataset.index < "2018-09-31")]['measure'].values
print(testing_set.shape)

validating_set = dataset[(dataset.index >= "2018-12-01") & (dataset.index < "2019-09-31")]['measure'].values
print(validating_set.shape)


#plt.plot(training_set, label = 'Shampoo Sales Data')
#plt.show()

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc = MinMaxScaler()
print('Creating training dataset')

whole_data = sc.fit_transform(whole_set.reshape(-1,1))
training_data = sc.fit_transform(training_set.reshape(-1,1))
testing_data = sc.fit_transform(testing_set.reshape(-1,1))
validating_data = sc.fit_transform(validating_set.reshape(-1,1))

seq_length = look_back
x, y = sliding_windows(whole_data, seq_length)
x_train, y_train = sliding_windows(training_data, seq_length)
x_test, y_test = sliding_windows(testing_data, seq_length)
x_val, y_val = sliding_windows(validating_data, seq_length)

# train_size = int(len(y) * 0.67)
# test_size = len(y) - train_size
#
dataX = Variable(torch.Tensor(np.array(x)).to(device))
dataY = Variable(torch.Tensor(np.array(y)).to(device))

trainX = Variable(torch.Tensor(np.array(x_train)).to(device))
trainY = Variable(torch.Tensor(np.array(y_train)).to(device))

testX = Variable(torch.Tensor(np.array(x_test)).to(device))
testY = Variable(torch.Tensor(np.array(y_test)).to(device))

valX = Variable(torch.Tensor(np.array(x_val)).to(device))
valY = Variable(torch.Tensor(np.array(y_val)).to(device))


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True).to(device)

        self.fc = nn.Linear(hidden_size, num_classes).to(device)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(device))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size).to(device)

        out = self.fc(h_out).to(device)

        return out




num_epochs = 50000
adam_learning_rate = 0.0001
sgd_learning_rate = 0.01

input_size = 1
hidden_size = hidden_size
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=adam_learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=sgd_learning_rate)

# To test during the training
dataY_plot = testY.data.cpu().numpy()
dataY_plot = sc.inverse_transform(dataY_plot)
actual_data_test = np.squeeze(dataY_plot)
test_data_length = len(actual_data_test)
test_during_training = True

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, trainY)

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        if epoch % 200 and test_during_training:
            train_predict = lstm(testX)
            data_predict = train_predict.data.cpu().numpy()
            data_predict = sc.inverse_transform(data_predict)
            #over_predicted_ratio = sum(actual_data_test < np.squeeze(data_predict)) / test_data_length
            diff = np.squeeze(data_predict) - actual_data_test
            over_predicted_val = np.sum(diff[diff > 0]) / np.sum(diff > 0)
            over_predicted_ratio = np.sum(diff > 0) / len(np.squeeze(data_predict))

            print('Epoch: {} - Overpredicted factor: {} - Overpredicted value: {}'.format(epoch, over_predicted_ratio, over_predicted_val))


lstm.eval()
train_predict = lstm(testX)

data_predict = train_predict.data.cpu().numpy()
dataY_plot = testY.data.cpu().numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)
df_result_2 = to_dataframe(np.squeeze(dataY_plot), np.squeeze(data_predict))
df_result_2['difference'] = df_result_2['actual'] - df_result_2['predicted']
df_result_2['mae'] = abs(df_result_2['actual'] - df_result_2['predicted'])
df_result_2['mse'] = df_result_2['difference']*df_result_2['difference']
df_result_2.to_csv(output_path + '/pre-MC.csv')

lstm.eval()
train_predict = lstm(valX)
data_predict = train_predict.data.cpu().numpy()
dataY_plot = valY.data.cpu().numpy()
data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)
df_result_3 = to_dataframe(np.squeeze(dataY_plot), np.squeeze(data_predict))
df_result_3['difference'] = df_result_3['actual'] - df_result_3['predicted']
df_result_3['mae'] = abs(df_result_3['actual'] - df_result_3['predicted'])
df_result_3['mse'] = df_result_3['difference'] * df_result_3['difference']
df_result_3.to_csv(output_path + '/post-MC.csv')



# plt.plot(dataY_plot)
# plt.plot(data_predict)
# plt.suptitle('Time-Series Prediction')
# plt.show()

