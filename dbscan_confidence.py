import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from collections import deque
import math

def get_data(file):
    ss = pd.read_csv(file)
    ss.storeId = ss.storeId.astype(int)
    k = ss[ss['storeId']==5]
    X = k[['NoOfSales', 'Outside', 'Inside']].values

    return X

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class DBSCAN(object):
    """
    DBSCAN clustering algorithm implementation
    """
    UNCLASSIFIED = -2
    NOISE = -1

    def __init__(self, eps=0.5, min_pts=5, severity_flag=1):
        """
        eps: epsilon value for epsilon-neighbourhood
        min_pts: number of points in neighbourhood to be considered as epsilon-neighbourhood
        """
        self.eps = eps
        self.min_pts = min_pts
        self.density = 0
        self.threshold = self.eps * min_pts
        self.severity_flag = severity_flag

        self.w1 = 66.8 / (66.8 + 27.2 + 4.2)
        self.w2 = 27.2 / (66.8 + 27.2 + 4.2)
        self.w3 = 4.2 / (66.8 + 27.2 + 4.2)

    def fit_predict(self, X):
        """
        X: data, numpy array of points
        return: array of cluster indices for points and noises
        """
        self.alldensity = [0] * X.shape[0]
        self.density = [0] * X.shape[0]
        self.severity = [0] * X.shape[0]

        labels = np.full([X.shape[0]], DBSCAN.UNCLASSIFIED)

        self.train_X = X
        return self._fit_predict(X, labels)

    def _fit_predict(self, X, labels):
        """
        Go through points in dataset and classify(set cluster id) them unless they are classified
        """
        cluster_id = DBSCAN._next_id(DBSCAN.NOISE)
        for point_ind in range(X.shape[0]):
            if labels[point_ind] == DBSCAN.UNCLASSIFIED and self._expand_cluster(X, labels, point_ind, cluster_id):
                # if self._expand_cluster(X, labels, point_ind, cluster_id):
                cluster_id = DBSCAN._next_id(cluster_id)
        outliers_index = [i for i in range(len(labels)) if labels[i] == -1]

        if self.severity_flag == 1:
            for i in outliers_index:
                d = np.sqrt(np.sum((X - X[i]) ** 2, axis=1)).tolist()
                dist_list = list(filter(lambda i: i != 0 and i < self.threshold, d))
                self.severity[i] = 0
                for dist in dist_list:
                    index = d.index(dist)
                    if labels[index] == -1:
                        continue
                    dist_severity = self.f_dist(dist, self.threshold)
                    density_severity = self.f_m(self.alldensity[index] + 1)
                    self.severity[i] += dist_severity * density_severity

                self.severity[i] = self.severity[i] / len(dist_list) if len(dist_list) != 0 else 0
                # self.severity[i] = (self.severity[i] - 1 / self.min_pts ** 2) / (1 - 1/self.min_pts ** 2)
                self.severity[i] = 1 - self.severity[i]
            return labels, self.severity
        else:
            return labels

    def predict(self, Y):
        self.predict_alldensity = [0] * Y.shape[0]
        self.predict_density = [0] * Y.shape[0]
        self.predict_severity = [0] * Y.shape[0]

        labels = np.full([Y.shape[0]], DBSCAN.UNCLASSIFIED)
        return self._predict_fit_predict(Y, labels)

    def _predict_fit_predict(self, X, labels):
        """
        Go through points in dataset and classify(set cluster id) them unless they are classified
        """
        cluster_id = DBSCAN._next_id(DBSCAN.NOISE)
        for point_ind in range(X.shape[0]):
            if labels[point_ind] == DBSCAN.UNCLASSIFIED and self._expand_cluster(self.train_X, labels, point_ind, cluster_id, predict_set = X):
                # if self._expand_cluster(X, labels, point_ind, cluster_id):
                cluster_id = DBSCAN._next_id(cluster_id)

        outliers_index = [i for i in range(len(labels)) if labels[i] == -1]

        if self.severity_flag == 1:
            for i in outliers_index:
                d = np.sqrt(np.sum((X - X[i]) ** 2, axis=1)).tolist()
                dist_list = list(filter(lambda i: i != 0 and i < self.threshold, d))
                self.predict_severity[i] = 0
                for dist in dist_list:
                    index = d.index(dist)
                    if labels[index] == -1:
                        continue
                    dist_severity = self.f_dist(dist, self.threshold)
                    density_severity = self.f_m(self.predict_alldensity[index] + 1)
                    self.predict_severity[i] += dist_severity * density_severity

                self.predict_severity[i] = self.predict_severity[i] / len(dist_list) if len(dist_list) != 0 else 0
                # self.severity[i] = (self.severity[i] - 1 / self.min_pts ** 2) / (1 - 1/self.min_pts ** 2)
                self.predict_severity[i] = 1 - self.predict_severity[i]
            return labels, self.predict_severity
        else:
            return labels

    def f_m(self, m):
        if m < self.min_pts:
            return float(m / self.min_pts)
        else:
            return 1

    def f_dist(self, dist, bound):
        if dist < bound / self.min_pts:
            return 1
        elif dist < bound:
            return 1 - dist / bound + 1 / self.min_pts
        else:
            return 1 / self.min_pts

    def _expand_cluster(self, X, labels, point_ind, cluster_id, predict_set=[]):
        """
        Add points to given cluster(cluster_id) if they are directly density reachable to given point(point_ind)
        """
        # find epsilon-neighbourhood of given point and add them to given cluster
        # otherwise classify them as noise

        if len(predict_set) != 0:
            region_inds = self._region_query(X, point_ind, predict_set)
            self.predict_alldensity[point_ind] = len(region_inds)
        else:
            region_inds = self._region_query(X, point_ind)
            self.alldensity[point_ind] = len(region_inds)

        if len(region_inds) < self.min_pts:
            labels[point_ind] = DBSCAN.NOISE
            return False

        if len(predict_set) != 0:
            self.predict_density[point_ind] = len(region_inds)
            labels[point_ind] = cluster_id
            return True
        else:
            self.density[point_ind] = len(region_inds)

        # label these points to cluseter_id
            labels[region_inds] = cluster_id
            labels[point_ind] = cluster_id

        # create queue of points in the epsilon-neighbourhood
        # consider all points in the queue to possibly belong to cluster
        # detemine it testing direct density reachability
        queue_inds = deque(region_inds)
        while len(queue_inds):
            current_point_ind = queue_inds.popleft()

            if len(predict_set) != 0:
                result = self._region_query(X, point_ind, predict_set)
                self.predict_alldensity[current_point_ind] = len(result)
            else:
                result = self._region_query(X, point_ind)
                self.alldensity[current_point_ind] = len(result)

            # test density reachability
            # on positive - consider only unclassified points and noises

            if len(result) > self.min_pts:
                is_noise = labels[result] == DBSCAN.NOISE
                is_unclassified = labels[result] == DBSCAN.UNCLASSIFIED
                # add only unclassified points
                queue_inds.extend(result[is_unclassified])
                # label these points to cluster_id
                labels[result[np.logical_or(is_noise, is_unclassified)]] = cluster_id
                if len(predict_set) != 0:
                    self.predict_density[current_point_ind] = len(result)
                else:
                    self.density[current_point_ind] = len(result)
        return True

    def _region_query(self, X, point_ind, predict_set=[]):
        """
        Find epsilon-neighbourhood of a given point(point_ind)
        """
        if len(predict_set) != 0:
            d = np.sqrt(np.sum((X - predict_set[point_ind]) ** 2, axis=1))
        else:
            d = np.sqrt(np.sum((X - X[point_ind]) ** 2, axis=1))
        mask = d < self.eps
        # mask[point_ind] = False  # exclude this point
        return np.where(mask)[0]

    @staticmethod
    def _next_id(cluster_id):
        return cluster_id + 1


def main():
    X= get_data('2018-12-07preproData.csv')
    scaler = MinMaxScaler()
    X1 = scaler.fit_transform(X)

    eps = 0.1
    db = DBSCAN(eps=eps, min_pts=5)
    labels, severity = db.fit_predict(X1)

    a = {}
    b = {}
    for i in range(len(labels)):
        if labels[i] == -1:
            a[i] = severity[i]
        else:
            b[i] = severity[i]

    print(X[:][labels==-1].shape)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels)
    ax.set_xlabel('Sales')
    ax.set_ylabel('Outside')
    ax.set_zlabel('Inside')
    ax.azim = 300
    plt.show()


if __name__ == '__main__':
    main()
