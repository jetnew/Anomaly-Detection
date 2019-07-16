from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt


class LSTM_Autoencoder:
    def __init__(self, optimizer='adam', loss='mse'):
        self.optimizer = optimizer
        self.loss = loss
        self.n_features = 1

    def build_model(self):
        timesteps = self.timesteps
        n_features = self.n_features
        model = Sequential()

        # Encoder
        model.add(LSTM(timesteps, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(LSTM(1, activation='relu'))
        model.add(RepeatVector(timesteps))

        # Decoder
        model.add(LSTM(timesteps, activation='relu', return_sequences=True))
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(n_features)))

        model.compile(optimizer=self.optimizer, loss=self.loss)
        model.summary()
        self.model = model

    def fit(self, X, epochs=3, batch_size=32):
        self.timesteps = X.shape[1]
        self.build_model()

        input_X = np.expand_dims(X, axis=2)
        self.model.fit(input_X, input_X, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        input_X = np.expand_dims(X, axis=2)
        output_X = self.model.predict(input_X)
        reconstruction = np.squeeze(output_X)
        return np.linalg.norm(X - reconstruction, axis=-1)

    def plot(self, scores, timeseries, threshold=0.95):
        sorted_scores = sorted(scores)
        threshold_score = sorted_scores[round(len(scores) * threshold)]

        plt.title("Reconstruction Error")
        plt.plot(scores)
        plt.plot([threshold_score] * len(scores), c='r')
        plt.show()

        anomalous = np.where(scores > threshold_score)
        normal = np.where(scores <= threshold_score)

        plt.title("Anomalies")
        plt.scatter(normal, timeseries[normal][:, -1], s=3)
        plt.scatter(anomalous, timeseries[anomalous][:, -1], s=5, c='r')
        plt.show()
