"""
1. 载入数据
2. 数据预处理
3. 划分新类数据
"""

from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix
from collections import Counter
import numpy as np
import warnings

warnings.filterwarnings("ignore")

"""
先划分数据集，再归一化与降维
"""
def load_data(standardize=True, reduction=True):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print("训练集共有{}个样本".format(x_train.shape[0]))
    print("测试集共有{}个样本".format(x_test.shape[0]))
    print("训练集共有{}个类别".format(len(np.unique(y_train))))

    # 归一化
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 展开
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # 标准化
    if standardize:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # PCA降维
    if reduction:
        pca = PCA(n_components=0.95)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    # 对训练集排序
    print("对训练集排序")
    train = np.c_[x_train, y_train]
    train = train[np.argsort(train[:,-1]), :]

    x_train = train[:, :-1]
    y_train = train[:, -1]

    #print(Counter(y_train))

    return x_train, y_train, x_test, y_test

def novelty_detection():
    x_train, y_train, x_test, y_test = load_data()

    num_per_class = int(x_train.shape[0] / len(np.unique(y_train)))
    num_known_classes = 7       # 已知类的个数
    known = np.array([0, 1, 2, 3, 4, 5, 6])     # 测试集中余下的类别作为新类

    num_train = num_per_class * num_known_classes
    x_train = x_train[:num_train]
    #y_train = y_train[:num_train]
    #y_train = int(known.__contains__(y_train))
    #y_test_new = int(known.__contains__(y_test))
    y_test = y_test.astype(np.int32).copy()
    #print(y_test)
    y_test[y_test <= 6] = 1
    y_test[y_test > 6] = -1
    #y_test[np.where(y_test==0)] = -1
    #print(np.unique(y_train))
    print(np.unique(y_test))
    #print(y_test)

    # 用LOF做新颖点检测
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1)
    print("-----fiting 训练集-----")
    lof.fit(x_train)
    print("-----预测测试集-----")
    y_pred = lof.predict(x_test)
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    novelty_detection()

# 结论：至少对于Fashion Mnist这个数据集而言，使用LOF进行新类检测几乎是无效的
