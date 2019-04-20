from keras.layers import LSTM, Dense
from keras.models import Sequential
import numpy as np


class LSTMAutoEncoder(object):
    def __init__(self):
        self.model = None

    def create_model(self, window_size):
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(window_size, 1), return_sequences=False))
        model.add(Dense(units=window_size, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        print(model.summary())
        return model

    def fit(self, timeseries, batch_size=8, epochs=5, validation_split=0.2, verbose=1):
        self.window_size = timeseries.shape[1]
        input_timeseries = np.expand_dims(timeseries, axis=2)
        self.model = self.create_model(self.window_size)
        self.model.fit(x=input_timeseries, y=timeseries,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_split=validation_split)

    def predict(self, timeseries):
        input_timeseries = np.expand_dims(timeseries, axis=2)
        output_timeseries = self.model.predict(x=input_timeseries)
        dist = np.linalg.norm(timeseries - output_timeseries, axis=-1)
        return dist