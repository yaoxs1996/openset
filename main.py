from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split

def generate_data():
    blob_centers = np.array(
        [[0.2, 2.3],
         [-1.5, 2.3],
         [-2.8, 1.8],
         [-2.8, 2.8],
         [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    x, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=42)

    return x, y

def plot_cluster(x, y=None, plot=True):
    plt.scatter(x[:, 0], x[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    if plot == True:
        plt.show()

def detect_outlier(x, n_neighbors=5):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    result = lof.fit_predict(x)
    print("There are", sum(result == -1), "outliers")
    new_x = np.c_[x, result]
    filter = np.where(new_x[:, 2]==-1)
    new_x = new_x[filter]
    
    return new_x

def df_to_ds(data):
    label = data[:, -1].astype(int)
    data = data[:, :-1]
    ds = tf.data.Dataset.from_tensor_slices((data, label))
    return ds

def train_test_val(x, y):
    #ds = df_to_ds(x, y)
    ds = np.c_[x, y]
    train_ds, test_ds = train_test_split(ds, test_size=0.2, random_state=42)
    train_ds, val_ds = train_test_split(train_ds, test_size=0.2, random_state=42)

    #print(val_ds)

    train_ds = df_to_ds(train_ds)
    test_ds = df_to_ds(test_ds)
    val_ds = df_to_ds(val_ds)

    return train_ds, test_ds, val_ds

def classify_model(x):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=x.shape[1:]))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(5, activation="softmax"))

    return model

def train(train, test, val):
    model = classify_model()
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=10)
    print(model.evaluate(test))

if __name__ == "__main__":
    x, y = generate_data()

    #train_ds, test_ds, val_ds = train_test_val(x, y)
    '''
    for ele in val_ds.as_numpy_iterator():
        print(ele)
    '''
    #y = tf.keras.utils.to_categorical(y, num_classes=5)
    #model.summary()
    #train(model, train_ds, test_ds, val_ds)
    model = classify_model(x)
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.fit(x, y, epochs=10)
