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

seq_length = 4
x, y = sliding_windows(whole_data, seq_length)
x_train, y_train = sliding_windows(training_data, seq_length)
x_test, y_test = sliding_windows(testing_data, seq_length)
x_val, y_val = sliding_windows(validating_data, seq_length)

# train_size = int(len(y) * 0.67)
# test_size = len(y) - train_size
#
dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x_train)))
trainY = Variable(torch.Tensor(np.array(y_train)))

testX = Variable(torch.Tensor(np.array(x_test)))
testY = Variable(torch.Tensor(np.array(y_test)))


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out

num_epochs = 200
learning_rate = 0.01

input_size = 1
hidden_size = 2
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

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



lstm.eval()
train_predict = lstm(testX)

data_predict = train_predict.data.numpy()
dataY_plot = testY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)
print(data_predict)
print(dataY_plot)
df_result_2 = to_dataframe(np.squeeze(dataY_plot), np.squeeze(data_predict))
print(df_result_2)
df_result_2['difference'] = df_result_2['actual'] - df_result_2['predicted']
diff_2 = sum(df_result_2['difference'])/len(df_result_2['difference'])
df_result_2.to_csv('post-MC.csv')
print("Test Post-MC diff %.6f" % diff_2)
#plt.axvline(x=training_set.shape[0], c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()

