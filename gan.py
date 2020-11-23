import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.python.ops.gen_math_ops import cross

BATCH_SIZE = 256
noise_dim = 100

def make_generator(input):
    model = keras.Sequential()
    model.add(Dense(128, use_bias=False, input_shape=(100,)))
    model.add(Dense(256, use_bias=False))
    model.add(Dense(input.shape[1]))

    assert model.output_shape == (None, input.shape[1])

    return model

def make_discriminator(input):
    model = keras.Sequential()
    model.add(InputLayer(input_shape=input.shape[1:]))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(1))

    return model

def get_cross_entropy():
    cross_entropy = losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy

def discriminator_loss(real_output, fake_output):
    cross_entropy = get_cross_entropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output):
    cross_entropy = get_cross_entropy()

    return cross_entropy(tf.ones_like(fake_output), fake_output)

def get_optimizers():
    generator_optimizer = optimizers.Adam(1e-4)
    discriminator_optimizer = optimizers.Adam(1e-4)

    return generator_optimizer, discriminator_optimizer

generator = make_generator(x_train)
discriminator = make_discriminator(x_train)

generator_optimizer, discriminator_optimizer = get_optimizers()

@tf.function
def train_step(x_train):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(x_train, training=True)
        fake_output = discriminator(generated_data, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for data_batch in dataset:
            train_step(data_batch)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time()-start))
