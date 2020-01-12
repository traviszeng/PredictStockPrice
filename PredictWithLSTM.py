# description: this uses an artificial recurrent neural network called Long Short Term Memory (LSTM).
#              to predict the closing stock price of a corporation using the past 60 days' stock price.
import pandas_datareader as web
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
import warnings

warnings.filterwarnings('ignore')

# get the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-01-10')
print(df)

# get the number of rows and columns in the dataset
print(df.shape)

# visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.show()

# create a new dataframe with only the Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

print(training_data_len)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))  # 0-1 range
scaled_data = scaler.fit_transform(dataset)

print(scaled_data)

# Create the training dataset
# Create the scaled training dataset
train_data = scaled_data[0:training_data_len, :]
# Split data
x_train = []
y_train = []

for i in range(60, len(train_data)):
    # 将最近60天的数据作为x
    x_train.append(train_data[i - 60:i, 0])
    # 将第60天的股价作为y
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()

# Convect the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data LSTM expects three dimensional data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

# Build the LSTM model
model = Sequential()
# Add LSTM layer
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing dataset
# Create a new array containing scaled values
test_data = scaled_data[training_data_len - 60:, :]
# Create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# Convert the data to a numpy array
x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
# inverse transform
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) **2 )
print(rmse)

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]

valid['Predictions'] = predictions
#Visualize the model
plt.figure(figsize=(16,8))
plt.title('Model performance')
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Close price USD',fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc = 'lower right')
plt.show()

print(valid)


# predict new val
# get the stock quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-01-10')
#Create new dataframe
new_df = apple_quote.filter(['Close'])
#Get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 dats
X_test.append(last_60_days_scaled)
#Convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#Get the prediction
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

