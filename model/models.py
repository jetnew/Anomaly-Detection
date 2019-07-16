from keras.layers import LSTM, Dense, Lambda, Input
from keras.models import Sequential, Model
from keras import backend as K
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


class VariationalAutoEncoder(object):
    def __init__(self):
        self.model = None

    def create_model(self, window_size):
        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

    def build_encoder(self, input_shape, intermediate_dim, latent_dim):
        def sampling(args):
            """Sample using given mean and variance."""
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean = 0 and std = 1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
