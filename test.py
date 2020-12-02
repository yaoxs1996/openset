from json import load
#import gan
#import data_loader
import numpy as np
from tensorflow.keras import losses, optimizers
from tensorflow.keras.models import load_model
from collections import Counter
from utils.CluStream import CluStream
from sklearn.cluster import KMeans
import math

def data_process():
    x_train, y_train, x_test, y_test = data_loader.load_data()
    y_train = y_train.astype(np.int).copy()
    y_test = y_test.astype(np.int).copy()

    y_train[y_train <= 6] = 1
    y_train[y_train > 6] = 0

    y_test[y_test <= 6] = 1
    y_test[y_test > 6] = 0

    return x_train, y_train, x_test, y_test

def model():
    x_train, y_train, x_test, y_test = data_process()
    discminator = gan.make_discriminator(x_train)
    discminator.compile(optimizer=optimizers.Adam(1e-4), loss=losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    discminator.fit(x_train, y_train, epochs=10)
    discminator.evaluate(x_test, y_test)

class MicroCluster():
    def __init__(self, x, y, cluster_center):
        self.linear_sum = x.sum(axis=0)
        self.square_sum = np.sum(x**2, axis=0)
        self.cluster_center = cluster_center

        # 计算微簇半径
        n_points = x.shape[0]
        ls_mean = self.linear_sum / n_points
        ss_mean = self.square_sum / n_points
        variance = ss_mean - ls_mean**2
        self.cluster_radius = math.sqrt(np.sum(variance))

        # 类别的相关信息：线性和、平方和、个数、半径、中心

def init():
    x_train = np.load("./data/x_train.npy")
    y_train = np.load("./data/y_train.npy")
    x_test = np.load("./data/x_test.npy")
    y_test = np.load("./data/y_test.npy")

    x_train = x_train[:7*6000, :]
    y_train = y_train[:7*6000]

    kmeans = KMeans(n_clusters=50, n_init=5, random_state=42, n_jobs=-1)
    kmeans.fit(x_train)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    micro_clusters = []

    for i in range(len(labels)):
        x = x_train[labels==i, :]
        y = y_train[labels==i]
        cluster_center = cluster_centers[i, :]

        temp = MicroCluster(x, y, cluster_center)
        micro_clusters.append(temp)

if __name__ == "__main__":
    init()
