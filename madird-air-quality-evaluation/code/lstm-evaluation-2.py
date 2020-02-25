# https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca
# https://romanorac.github.io/machine/learning/2019/09/27/time-series-prediction-with-lstm.html
import sys
import torch
import torch.nn as nn
import os

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path


from sklearn.preprocessing import MinMaxScaler


data_txt = '../data/txt/'
data_csv = '../data/csv/'
results_folder = '../data/results/'

if len(sys.argv) < 2:
    print('Error: Output folder prefix is needed')
    sys.exit(0)
elif os.path.exists(results_folder+sys.argv[1]):
    print('Error: Output folder {} already exists'.format(results_folder+sys.argv[1]))
    sys.exit(0)
else:
    output_folder_prefix = sys.argv[1]
    os.mkdir(results_folder+'/'+output_folder_prefix)


dataset = pd.read_csv(data_csv+'035-08-sequence_air_all.csv', header=0, index_col=0)

dataset.replace(-1, dataset.mean())

all_data = dataset['measure'].values.astype(float)

all_data = dataset[(dataset.index >= "2013-12-01") & (dataset.index <= "2019-11-30")]['measure'].values.astype(float)

df_train = dataset[dataset.index < "2017-12-01"]['measure'].to_frame(name='NO2')
print(df_train.shape)

df_val = dataset[(dataset.index >= "2017-12-01") & (dataset.index < "2018-09-31")]['measure'].to_frame(name='NO2')
print(df_val.shape)

df_test = dataset[(dataset.index >= "2018-12-01") & (dataset.index < "2019-09-31")]['measure'].to_frame(name='NO2')
print(df_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_arr = scaler.fit_transform(df_train)
val_arr = scaler.transform(df_val)
test_arr = scaler.transform(df_test)


def transform_data(arr, seq_len):
    x, y = [], []
    for i in range(len(arr) - seq_len):
        x_i = arr[i : i + seq_len]
        y_i = arr[i + 1 : i + seq_len + 1]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var

from torch.autograd import Variable

seq_len = 72

x_train, y_train = transform_data(train_arr, seq_len)
x_val, y_val = transform_data(val_arr, seq_len)
x_test, y_test = transform_data(test_arr, seq_len)



def plot_sequence(axes, i, x_train, y_train):
    axes[i].set_title("%d. Sequence" % (i + 1))
    axes[i].set_xlabel("Time Bars")
    axes[i].set_ylabel("Scaled VWAP")
    axes[i].plot(range(seq_len), x_train[i].cpu().numpy(), color="r", label="Feature")
    axes[i].plot(range(1, seq_len + 1), y_train[i].cpu().numpy(), color="b", label="Target")
    axes[i].legend()


# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
# plot_sequence(axes, 0, x_train, y_train)
# plot_sequence(axes, 1, x_train, y_train)
# plt.show()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")




import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size).to(device)
        self.linear = nn.Linear(self.hidden_size, self.output_size).to(device)

    def forward(self, input, future=0, y=None):
        outputs = []

        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(device)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32).to(device)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t).to(device)
            outputs += [output]

        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


import time
import random


class Optimization:
    """ A helper class to train, test and diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []

    @staticmethod
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=72,
        n_epochs=0,
        do_teacher_forcing=None,
    ):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []

            train_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train, batch_size):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = self._predict(x_batch, y_batch, seq_len, do_teacher_forcing)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()
            train_loss /= batch
            self.train_losses.append(train_loss)

            self._validation(x_val, y_val, batch_size)

            elapsed = time.time() - start_time
            print(
                "Epoch %d Train loss: %.6f. Validation loss: %.6f. Avg future: %.2f. Elapsed time: %.2fs."
                % (epoch + 1, train_loss, self.val_losses[-1], np.average(self.futures), elapsed)
            )

    def _predict(self, x_batch, y_batch, seq_len, do_teacher_forcing):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if do_teacher_forcing:
            future = random.randint(1, int(seq_len) / 2)
            limit = x_batch.size(1) - future
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit:])
        else:
            future = 0
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred

    def _validation(self, x_val, y_val, batch_size):
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        if x_val is None or y_val is None:
            return
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val, batch_size):
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                val_loss += loss.item()
            val_loss /= batch
            self.val_losses.append(val_loss)

    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = self.model(x_batch, future=future)
                #y_pred = (
                #    y_pred[:, -len(y_batch) :] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                #)
                print(x_batch.shape[1])
                print(y_batch.shape[1])
                print(batch)
                if y_pred.shape[1] > y_batch.shape[1]:
                    y_pred = y_pred[:, -len(y_batch):]

                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch
            return actual, predicted, test_loss

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")


def generate_sequence(scaler, model, x_sample, future=1000):
    """ Generate future values for x_sample with the model """
    y_pred_tensor = model(x_sample, future=future)
    y_pred = y_pred_tensor.cpu().tolist()
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred


def to_dataframe(actual, predicted):
    return pd.DataFrame({"actual": actual, "predicted": predicted})


def inverse_transform(scalar, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


# model_1 = Model(input_size=1, hidden_size=21, output_size=1)
# loss_fn_1 = nn.MSELoss()
# optimizer_1 = optim.Adam(model_1.parameters(), lr=1e-3)
# scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=5, gamma=0.1)
# optimization_1 = Optimization(model_1, loss_fn_1, optimizer_1, scheduler_1)
#
#
# optimization_1.train(x_train, y_train, x_val, y_val, do_teacher_forcing=False)
# optimization_1.plot_losses()
#
# actual_1, predicted_1, test_loss_1 = optimization_1.evaluate(x_test, y_test, future=5, batch_size=100)
# df_result_1 = to_dataframe(actual_1, predicted_1)
# df_result_1 = inverse_transform(scaler, df_result_1, ['actual', 'predicted'])
# df_result_1.plot(figsize=(14, 7))
# print("Test loss %.6f" % test_loss_1)
#
# x_sample = x_test[0].reshape(1, -1)
# y_sample = df_test.vwap[:1100]
#
# y_pred1 = generate_sequence(scaler, optimization_1.model, x_sample)
# plt.figure(figsize=(14, 7))
# plt.plot(range(100), y_pred1[0][:100], color="blue", lw=2, label="Predicted VWAP")
# plt.plot(range(100, 1100), y_pred1[0][100:], "--", color="blue", lw=2, label="Generated VWAP")
# plt.plot(range(0, 1100), y_sample, color="red", label="Actual VWAP")
# plt.legend()
#


model_2 = Model(input_size=1, hidden_size=50, output_size=1)
loss_fn_2 = nn.MSELoss()
optimizer_2 = optim.Adam(model_2.parameters(), lr=1e-3)
scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size=5, gamma=0.1)
optimization_2 = Optimization(model_2, loss_fn_2,  optimizer_2, scheduler_2)
optimization_2.train(x_train, y_train, x_val, y_val, batch_size=72, n_epochs=0, do_teacher_forcing=True)
optimization_2.plot_losses()
actual_3, predicted_3, test_loss_3 = optimization_2.evaluate(x_val, y_val, batch_size=72, future=5)
df_result_3 = to_dataframe(actual_3, predicted_3)
df_result_3 = inverse_transform(scaler, df_result_3, ["actual", "predicted"])
df_result_3.plot(figsize=(14, 7))
df_result_3['difference'] = df_result_3['actual'] - df_result_3['predicted']
df_result_3.to_csv(results_folder + '/' + output_folder_prefix + '/pre-MC.csv')
diff_3 = sum(df_result_3['difference'])/len(df_result_3['difference'])
print("Test PRE-MC loss %.6f" % test_loss_3)
print("Test PRE-MC diff %.6f" % diff_3)


actual_2, predicted_2, test_loss_2 = optimization_2.evaluate(x_test, y_test, batch_size=72, future=5)
df_result_2 = to_dataframe(actual_2, predicted_2)
df_result_2 = inverse_transform(scaler, df_result_2, ["actual", "predicted"])
df_result_2.plot(figsize=(14, 7))
df_result_2['difference'] = df_result_2['actual'] - df_result_2['predicted']
df_result_2.to_csv(results_folder + '/' + output_folder_prefix + '/post-MC.csv')
diff_2 = sum(df_result_2['difference'])/len(df_result_2['difference'])
print("Test Post-MC loss %.6f" % test_loss_2)
print("Test Post-MC diff %.6f" % diff_2)


#plt.show()



