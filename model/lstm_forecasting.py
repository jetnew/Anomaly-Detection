
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

window_size = 10
timesteps = window_size-1
n_features = 1

model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
model.add(LSTM(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')