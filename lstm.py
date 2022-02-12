import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, r2_score

tickers = ['AAPL']

# 'NVDA', 'MSFT', 'FB', 'RBLX', 'LCID', 'BABA', 'GOOGL', 'PYPL', 'V', 'DIS', 'FB', 'MSFT', 'GOOGL', 'V', 'DIS'
start = dt.datetime(2018, 1, 1)
end = dt.datetime(2020, 1, 1)

data = []


for t in tickers:
  data.append(web.DataReader(t, 'yahoo', start, end))

aapl = web.DataReader('AAPL', 'yahoo', start, end)
p_days = [45, 60]
e = [15, 20, 25]
b = [8, 16, 24]
units = [30, 40, 50]
dropout = [0.15, 0.2, 0.25]

# min_rms = 1000
# result = []
# for i in p_days:
#   for j in e:
#     for k in b:
#       for l in units:
#         for m in dropout: 
#           r, r2 = stock_prediction(0, aapl, l, m, i, j, k)
#           result.append((r, r2, i, j, k, l, m))
#           print('rms: {}\tr2: {}\tdays: {}\tepochs: {}\tbatch_size:{}\tunits: {}\tdropout_rate: {} '.format(r, r2, i, j, k, l, m ))

# for i in result:
#   print('rms: {}\tr2: {}\tdays: {}\tepochs: {}\tbatch_size:{} '.format(i[0], i[1],i[2], i[3], i[4]))

r, r2 = stock_prediction(0, aapl, 50, 0.15, 45, 25, 16)
r, r2 = stock_prediction(0, aapl, 50, 0.15, 45, 20, 8)
r, r2 = stock_prediction(0, aapl, 50, 0.15, 45, 20, 16)
r, r2 = stock_prediction(0, aapl, 50, 0.2, 60, 25, 16)
# for index, val in enumerate(data):
#   for i in units:
#     for j in dropout:
#       r, r2 = stock_prediction(index, val, i, j, 60, 25, 32)
#       result.append((r, r2, i, j))

# for i in result:
#   print('rms: {}\tr2: {}\tunits: {}\tdropout:{} '.format(i[0], i[1] , i[2], i[3]))
def stock_prediction(index, data, units, dropout, prediction_days, epochs, batch_size):
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1,1))

  
  x_train = []
  y_train = []

  for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


  model = Sequential()
  model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(Dropout(dropout))
  model.add(LSTM(units=units, return_sequences=True))
  model.add(Dropout(dropout))
  model.add(LSTM(units=units))
  model.add(Dropout(dropout))
  model.add(Dense(units=1))

  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

  test_start = dt.datetime(2020, 1, 1)
  test_end = dt.datetime(2021, 11, 30)

  test_data = web.DataReader(tickers[index], 'yahoo', test_start, test_end)
  actual_prices = test_data['Adj Close'].values

  total_dataset = pd.concat((data['Adj Close'], test_data['Adj Close']), axis=0)

  model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
  model_inputs = model_inputs.reshape(-1, 1)
  model_inputs = scaler.transform(model_inputs)

  x_test = []

  for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  predicted_prices = model.predict(x_test)
  predicted_prices = scaler.inverse_transform(predicted_prices)
  # score = model.evaluate(predicted_prices, actual_prices, verbose=0)
  # print(score)
  p = []
  for i in predicted_prices:
    p.append(i[0])
  
  rms = mean_squared_error(actual_prices, p, squared=False)
  print(rms)
  r2 = r2_score(actual_prices, p)
  print(r2)


  company = tickers[index]
  plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
  plt.plot(predicted_prices, color="orange", label=f"Predicted {company} Price")
  plt.title(f"{company} Share Price")
  plt.xlabel('Time')
  plt.ylabel(f'{company} Share Price')
  plt.legend()
  plt.show()
  model.summary
  return rms, r2
  # real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
  # real_data = np.array(real_data)
  # real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

  # print(real_data, len(real_data))

  # prediction = model.predict(real_data)
  # prediction = scaler.inverse_transform(prediction)
  # print(f"Prediction: {prediction}")