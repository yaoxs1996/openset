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
    def __init__(self, x, y, cluster_center, class_labels):
        self.linear_sum = x.sum(axis=0)
        self.square_sum = np.sum(x**2, axis=0)
        self.cluster_center = cluster_center

        # 计算微簇半径
        # n_points = x.shape[0]
        # ls_mean = self.linear_sum / n_points
        # ss_mean = self.square_sum / n_points
        # variance = ss_mean - ls_mean**2
        # self.cluster_radius = math.sqrt(np.sum(variance))
        self.cluster_radius = self.__compute_radius(x, self.linear_sum, self.square_sum)

        # 类别的相关信息：线性和、平方和、个数、半径、中心
        self.class_labels = class_labels
        self.class_counts = self.__compute_counts(y)

        self.class_linear_sum = []
        self.class_square_sum = []
        self.class_centers = []
        self.class_radius = []

        for i in range(len(self.class_labels)):
            if self.class_counts[i] == 0:
                self.class_linear_sum.append(np.zeros(x.shape[1]))
                self.class_square_sum.append(np.zeros(x.shape[1]))
                self.class_centers.append(np.zeros(x.shape[1]))
                self.class_radius.append(0.0)
                continue

            index = (y==i)
            data = x[index, :]
            linear_sum = data.sum(axis=0)
            self.class_linear_sum.append(linear_sum)

            square_sum = np.sum(data**2, axis=0)
            self.class_square_sum.append(square_sum)

            center = self.__compute_center(data)
            self.class_centers.append(center)

            radius = self.__compute_radius(data, linear_sum, square_sum)
            self.class_radius.append(radius)

    def __compute_radius(self, data, ls, ss):
        n_points = data.shape[0]
        ls_mean = ls / n_points
        ss_mean = ss / n_points
        variance = ss_mean - ls_mean**2
        radius = math.sqrt(np.sum(variance))

        return radius

    def __compute_center(self, data):
        return np.mean(data, axis=0)

    def __compute_counts(self, y):
        class_counts = []
        for i in range(len(self.class_labels)):
            count = np.sum(y==i)
            class_counts.append(count)

        return class_counts

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
    train_labels = np.unique(y_train)

    for i in range(len(labels)):
        x = x_train[labels==i, :]
        y = y_train[labels==i]
        cluster_center = cluster_centers[i, :]

        temp = MicroCluster(x, y, cluster_center, train_labels)
        micro_clusters.append(temp)

if __name__ == "__main__":
    init()
