import math
import numpy as np

class MicroCluster:
    """
    实现 CluStream 算法的 MicroCluster 数据结构
    Parameters
    ----------
    :parameter nb_points 簇内点的个数
    :parameter identifier 簇的标识符（-1表示该簇来自两个簇的合并）
    :parameter merge 表示该簇是否是来自两个已有簇的合并的结果
    :parameter id_list 合并的簇的id列表
    :parameter linear_sum 簇内点的线性和
    :parameter squared_sum 加入到簇中所有点的平方和
    :parameter linear_time_sum 所有时间戳加入到簇中点的线性和
    :parameter squared_time_sum 所有时间戳加入到簇中点的平方和
    :parameter m 认为能够决定一个簇关联戳的点的个数
    :parameter update_timestamp 用于表示簇的最后更新时间
    """

    def __init__(self, nb_points=0, identifier=0, id_list=None, linear_sum=None,
                 squared_sum=None, linear_time_sum=0, squared_time_sum=0,
                 m=100, update_timestamp=0):
        self.nb_points = nb_points
        self.identifier = identifier
        self.id_list = id_list
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.linear_time_sum = linear_time_sum
        self.squared_time_sum = squared_time_sum
        self.m = m
        self.update_timestamp = update_timestamp
        self.radius_factor = 1.8
        self.epsilon = 0.00005
        self.min_variance = math.pow(1, -5)

    # 计算微簇中心
    def get_center(self):
        center = [self.linear_sum[i] / self.nb_points for i in range(len(self.linear_sum))]
        return center

    def get_weight(self):
        return self.nb_points

    def insert(self, new_point, current_timestamp):
        self.nb_points += 1
        self.update_timestamp = current_timestamp
        for i in range(len(new_point)):
            self.linear_sum[i] += new_point[i]
            self.squared_sum[i] += math.pow(new_point[i], 2)
            self.linear_time_sum += current_timestamp
            self.squared_time_sum += math.pow(current_timestamp, 2)

    def merge(self, micro_cluster):
        # 必须删除 micro_cluster
        self.nb_points += micro_cluster.nb_points
        self.linear_sum += micro_cluster.linear_sum
        self.squared_sum += micro_cluster.squared_sum
        self.linear_time_sum += micro_cluster.linear_time_sum
        self.squared_time_sum += micro_cluster.squared_time_sum

        if(self.identifier != -1):
            if(micro_cluster.identifier != -1):
                self.id_list = [self.identifier, micro_cluster.identifier]
            else:
                micro_cluster.id_list.append(self.identifier)
                self.id_list = micro_cluster.id_list.copy()
            self.identifier = -1
        else:
            if(micro_cluster.identifier != -1):
                self.id_list.append(micro_cluster.identifier)
            else:
                self.id_list.extend(micro_cluster.id_list)

    def get_relevancestamp(self):
        if(self.nb_points < 2*self.m):
            return self.get_mutime()
        return self.get_mutime() + self.get_sigmatime() * self.get_quantile(self.m / (2*self.nb_points))

    def get_mutime(self):
        return self.linear_time_sum / self.nb_points

    def get_sigmatime(self):
        return math.sqrt(self.squared_time_sum / self.nb_points - math.pow((self.linear_time_sum / self.nb_points), 2))

    def get_quantile(self, x):
        assert(x >= 0 and x <= 1)
        return math.sqrt(2) * self.inverse_error(2 * x - 1)

    def get_radius(self):
        if self.nb_points == 1:
            return 0
        return self.get_deviation() * self.radius_factor

    def get_cluster_feature(self):
        return self.this

    def get_deviation(self):
        variance = self.get_variance_vec()
        sum_deviation = 0
        for i in range(len(variance)):
            sqrt_deviation = math.sqrt(variance[i])
            sum_deviation += sqrt_deviation
        return sum_deviation / len(variance)

    def get_variance_vec(self):
        variance_vec = list()
        for i in range(len(self.linear_sum)):
            ls_mean = self.linear_sum[i] / self.nb_points
            ss_mean = self.squared_sum[i] / self.nb_points
            variance = ss_mean - math.pow(ls_mean, 2)
            if variance <= 0:
                if variance > -self.epsilon:
                    variance = self.min_variance

            variance_vec.append(variance)
        return variance_vec

    def inverse_error(self, x):
        z = (math.sqrt(math.pi) * x)
        inv_error = z / 2
        z_prod = math.pow(z, 3)
        inv_error += (1 / 24) * z_prod

        z_prod *= math.pow(z, 2)
        inv_error += (7 / 960) * z_prod

        z_prod = math.pow(z, 2)
        inv_error += (127 * z_prod) / 80640

        z_prod = math.pow(z, 2)
        inv_error += (4369 / z_prod) * 11612160

        z_prod = math.pow(z, 2)
        inv_error += (34807 / z_prod) * 364953600

        z_prod = math.pow(z, 2)
        inv_error += (20036983 / z_prod) * 0x797058662400d
        return z_prod