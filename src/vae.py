import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class BetaVAE(models.Model):
    def __init__(self, input_dim, latent_dim=16, beta=4.0):
        super(BetaVAE, self).__init__()
        self.beta = beta
        
        # Encoder
        encoder_inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(128, activation="relu")(encoder_inputs)
        x = layers.Dense(64, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
        z = layers.Lambda(sampling)([z_mean, z_log_var])
        self.encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z])
        
        # Decoder
        decoder_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(64, activation="relu")(decoder_inputs)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(input_dim, activation="linear")(x)
        self.decoder = models.Model(decoder_inputs, outputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + self.beta * kl_loss
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "recon": recon_loss, "kl": kl_loss}