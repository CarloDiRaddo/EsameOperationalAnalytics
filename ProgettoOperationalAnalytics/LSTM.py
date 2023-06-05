import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('final_Liquor_Sales.csv', header=0)
df['period'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
aSales = df['Sale (Dollars)'].to_numpy()
logdata = np.log(aSales)
data = pd.Series(logdata)
plt.rcParams["figure.figsize"] = (10, 8)
plt.plot(data.values)
plt.show()

# train and test set
train = data[:-12]
Test = data[-12:]
reconstruct = np.exp(np.r_[train, Test])

# ------------------------------------------------- neural forecast
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit_transform(train.values.reshape(-1, 1))
scaled_train_data = scaler.transform(train.values.reshape(-1, 1))
scaled_test_data = scaler.transform(Test.values.reshape(-1, 1))

from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 12
n_features = 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(20, activation='relu', input_shape=(n_input, n_features), dropout=0.05))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(generator, epochs=25)
lstm_model.summary()

losses_lstm = lstm_model.history.history['loss']
plt.xticks(np.arange(0, 21, 1))
plt.plot(range(len(losses_lstm)), losses_lstm)
plt.show()

lstm_predictions_scaled = list()
batch = scaled_train_data[-n_input:]
curbatch = batch.reshape((1, n_input, n_features))

for i in range(len(Test)):
    lstm_pred = lstm_model.predict(curbatch)[0]
    lstm_predictions_scaled.append(lstm_pred)
    curbatch = np.append(curbatch[:, 1:, :], [[lstm_pred]], axis=1)

lstm_forecast = scaler.inverse_transform(lstm_predictions_scaled)
yfore = np.transpose(lstm_forecast).squeeze()

# reconstruction
expdata = np.exp(train)
expfore = np.exp(yfore)
plt.plot(aSales, label="Sale (Dollars)")
plt.plot(expdata, label='Data')
plt.plot([None for _ in expdata] + [x for x in expfore], label='Forecast')
plt.legend()
plt.show()

# Calculate RMSE and MAE
reconstructed_sales = np.exp(np.concatenate([train.values, Test.values]))
rmse = np.sqrt(mean_squared_error(reconstructed_sales[-12:], expfore))
mae = mean_absolute_error(reconstructed_sales[-12:], expfore)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
