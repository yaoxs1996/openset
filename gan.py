from json import load
import tensorflow as tf
import time
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, Activation
from tensorflow.keras.models import load_model
from tensorflow import nn
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
import data_loader
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

BATCH_SIZE = 256
noise_dim = 100
num_classes = 7

# 生成器
# 输入为100维的噪声数据
def make_generator(input):
    model = keras.Sequential()
    model.add(Dense(256, use_bias=False, input_shape=(noise_dim,)))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    model.add(Dense(512, use_bias=False))
    # model.add(BatchNormalization())
    # model.add(Activation("relu"))
    model.add(Dense(input.shape[1], activation="sigmoid", use_bias=False))

    assert model.output_shape == (None, input.shape[1])

    return model

def make_discriminator(input):
    model = keras.Sequential()
    model.add(InputLayer(input_shape=input.shape[1:]))
    # 将relu替换成LeakyReLu
    model.add(Dense(256, activation="relu", use_bias=False))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(0.2))
    model.add(Dense(512, activation="relu", use_bias=False))
    # model.add(BatchNormalization())
    # model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation="sigmoid", use_bias=False))

    return model

# 增加一个分类器模型，仿照论文的方法进行实验
def make_classifier(x, y):
    n_classes = len(np.unique(y))
    model = keras.Sequential()
    model.add(InputLayer(input_shape=x.shape[1:]))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))

    return model

"""
def get_cross_entropy():
    cross_entropy = losses.BinaryCrossentropy(from_logits=True)

    return cross_entropy
"""

def discriminator_loss(real_output, fake_output):
    cross_entropy = losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output, kl_fake_output):
    cross_entropy = losses.BinaryCrossentropy(from_logits=True)

    unifom_dist = tf.ones([BATCH_SIZE, num_classes]) * (1.0 / num_classes)
    kl = losses.KLDivergence()
    kl_loss = kl(kl_fake_output, unifom_dist) * num_classes

    return cross_entropy(tf.ones_like(fake_output), fake_output) + kl_loss

def get_optimizers():
    generator_optimizer = optimizers.Adam(1e-4)
    discriminator_optimizer = optimizers.Adam(1e-4)

    return generator_optimizer, discriminator_optimizer

x_train, y_train, x_test, y_test = data_loader.load_data()
generator = make_generator(x_train)
discriminator = make_discriminator(x_train)
classifier = load_model("./models/classifier")

generator_optimizer, discriminator_optimizer = get_optimizers()

@tf.function
def train_step(x_train):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(x_train, training=True)
        fake_output = discriminator(generated_data, training=True)

        kl_fake_output = nn.log_softmax(classifier(generated_data))

        gen_loss = generator_loss(fake_output, kl_fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        #tf.print("g_loss: ", gen_loss, "d_loss: ", disc_loss)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    #tf.print("g_loss: ", gen_loss, "d_loss: ", disc_loss)
    #tf.print(type(gen_loss))
    # g_loss = gen_loss.numpy()
    # d_loss = disc_loss.numpy()
    # return g_loss, d_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for data_batch in dataset:
            #info = train_step(data_batch)
            train_step(data_batch)

        print("Time for epoch {} is {:.4f} sec".format(epoch + 1, time.time()-start))
        #print(info)

def test(x_test, y_test):
    #y_pred = discriminator.predict(x_test)
    y_pred = discriminator.predict_classes(x_test)
    print(y_pred)
    print(np.unique(y_pred))
    
    print("准确度为：")
    print(accuracy_score(y_test, y_pred))
    print("召回率为：")
    print(recall_score(y_test, y_pred))
    print("混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

def main():
    #x_train, y_train, x_test, y_test = data_loader.load_data()
    global x_train
    global y_train
    global x_test
    global y_test

    num_per_class = int(x_train.shape[0] / len(np.unique(y_train)))
    num_known_classes = 7       # 已知类的个数
    num_train = num_per_class * num_known_classes
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(100).batch(BATCH_SIZE)
    print("-----开始训练生成对抗网络-----")
    train(train_dataset, 100)

    # 划分新类：0-6为已知类，7-9为新类
    y_test = y_test.astype(np.int).copy()       # 原测试标签是只读的
    y_test[y_test <= 6] = 1
    y_test[y_test > 6] = 0

    print("-----开始测试新类检测-----")
    test(x_test, y_test)

if __name__ == "__main__":
    # x_train, _, _, _ = data_loader.load_data()
    # print(x_train.shape)
    # generator = make_generator(x_train)
    # noise = tf.random.normal([1, 100])
    # print(generator(noise))
    main()
