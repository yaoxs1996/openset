import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

def load_process_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # 归一化
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # 降维
    pca = PCA(n_components=0.95)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # 划分验证集
    x_train_pca, x_val_pca, y_train, y_val = train_test_split(x_train_pca, y_train, test_size=0.2)

    return x_train_pca, y_train, x_val_pca, y_val, x_test_pca, y_test

def create_model(input):
    model = keras.Sequential()

    model.add(keras.Input(shape=input.shape[1:]))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))

    return model

def train():
    x_train, y_train, x_test, y_test, x_val, y_val = load_process_data()
    model = create_model(x_train)

    optimizer = keras.optimizers.Adam(1e-3)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    print("-----训练阶段-----")
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
    print("-----测试阶段-----")
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    train()