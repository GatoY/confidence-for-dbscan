# -*- coding: utf-8 -*-
"""Ordering Points To Identify the Clustering Structure (OPTICS)

These routines execute the OPTICS algorithm, and implement various
cluster extraction methods of the ordered list.

Authors: Shane Grigsby <refuge@rocktalus.com>
         Amy X. Zhang <axz@mit.edu>
         Erich Schubert <erich@debian.org>
License: BSD 3 clause
"""
import warnings
import numpy as np

from sklearn.utils import check_array
from sklearn.utils import gen_batches, get_chunk_n_rows
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances

from sklearn.neighbors import NearestNeighbors


def optics(X, outlier_ratio=0.01, min_samples=5, max_eps=np.inf, metric='minkowski',
           p=2, metric_params=None, maxima_ratio=.75,
           rejection_ratio=.7, similarity_threshold=0.4,
           significant_min=.003, min_cluster_size=.005,
           min_maxima_ratio=0.001, algorithm='auto',
           leaf_size=30, n_jobs=None):
    clust = OPTICS(min_samples, max_eps, metric, p, metric_params,
                   maxima_ratio, rejection_ratio,
                   similarity_threshold, significant_min,
                   min_cluster_size, min_maxima_ratio,
                   algorithm, leaf_size, n_jobs)
    clust.fit(X)
    # use outlier_ratio to find the optimise x.
    #     x = optimise_x(clust.reachability_, outlier_ratio)
    #
    r = clust.reachability_
    r = r[np.where(r <= 1)]
    x = r.mean() * 10
    clust.run_dbscan(x)
    #     return clust
    print(x)
    return clust.reachability_, clust.labels_


def optimise_x(a, outlier_ratio, stop=0.1, alpha=0.1, max_round=1000):
    c = 0
    x = 0.05
    while (1):
        c += 1
        #         print('x:'+str(x))
        c = np.where(np.array(a) > x)[0].shape[0]
        F = c / len(a) - outlier_ratio
        #         print(' ')
        if abs(F) / outlier_ratio < stop:
            print('break')
            break
        x = x + alpha * F
        if c > max_round:
            print('reach max_round 1000 so break.')
            break
    return x


class OPTICS(BaseEstimator, ClusterMixin):

    def __init__(self, min_samples=5, max_eps=np.inf, metric='minkowski',
                 p=2, metric_params=None, maxima_ratio=.75,
                 rejection_ratio=.7, similarity_threshold=0.4,
                 significant_min=.003, min_cluster_size=.005,
                 min_maxima_ratio=0.001, algorithm='auto',
                 leaf_size=30, n_jobs=None):

        self.max_eps = max_eps
        self.min_samples = min_samples
        self.maxima_ratio = maxima_ratio
        self.rejection_ratio = rejection_ratio
        self.similarity_threshold = similarity_threshold
        self.significant_min = significant_min
        self.min_cluster_size = min_cluster_size
        self.min_maxima_ratio = min_maxima_ratio
        self.algorithm = algorithm
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float)

        n_samples = len(X)

        if self.min_samples > n_samples:
            raise ValueError("Number of training samples (n_samples=%d) must "
                             "be greater than min_samples (min_samples=%d) "
                             "used for clustering." %
                             (n_samples, self.min_samples))

        if self.min_cluster_size <= 0 or (self.min_cluster_size !=
                                          int(self.min_cluster_size)
                                          and self.min_cluster_size > 1):
            raise ValueError('min_cluster_size must be a positive integer or '
                             'a float between 0 and 1. Got %r' %
                             self.min_cluster_size)
        elif self.min_cluster_size > n_samples:
            raise ValueError('min_cluster_size must be no greater than the '
                             'number of samples (%d). Got %d' %
                             (n_samples, self.min_cluster_size))

        # Start all points as 'unprocessed' ##
        self.reachability_ = np.empty(n_samples)
        self.reachability_.fill(np.inf)
        self.predecessor_ = np.empty(n_samples, dtype=int)
        self.predecessor_.fill(-1)
        # Start all points as noise ##
        self.labels_ = np.full(n_samples, -1, dtype=int)

        nbrs = NearestNeighbors(n_neighbors=self.min_samples,
                                algorithm=self.algorithm,
                                leaf_size=self.leaf_size, metric=self.metric,
                                metric_params=self.metric_params, p=self.p,
                                n_jobs=self.n_jobs)

        nbrs.fit(X)
        # Here we first do a kNN query for each point, this differs from
        # the original OPTICS that only used epsilon range queries.
        self.core_distances_ = self._compute_core_distances_(X, nbrs)
        # OPTICS puts an upper limit on these, use inf for undefined.
        self.core_distances_[self.core_distances_ > self.max_eps] = np.inf
        self.ordering_ = self._calculate_optics_order(X, nbrs)

        #         indices_, self.labels_ = _extract_optics(self.ordering_,
        #                                                  self.reachability_,
        #                                                  self.maxima_ratio,
        #                                                  self.rejection_ratio,
        #                                                  self.similarity_threshold,
        #                                                  self.significant_min,
        #                                                  self.min_cluster_size,
        #                                                  self.min_maxima_ratio)

        return self

    def run_dbscan(self, eps):
        indices_, self.labels_ = self.extract_dbscan(eps=eps)
        self.core_sample_indices_ = indices_

    # OPTICS helper functions
    def _compute_core_distances_(self, X, neighbors, working_memory=None):

        n_samples = len(X)
        core_distances = np.empty(n_samples)
        core_distances.fill(np.nan)

        chunk_n_rows = get_chunk_n_rows(row_bytes=16 * self.min_samples,
                                        max_n_rows=n_samples,
                                        working_memory=working_memory)
        slices = gen_batches(n_samples, chunk_n_rows)
        for sl in slices:
            core_distances[sl] = neighbors.kneighbors(
                X[sl], self.min_samples)[0][:, -1]
        return core_distances

    def _calculate_optics_order(self, X, nbrs):
        # Main OPTICS loop. Not parallelizable. The order that entries are
        # written to the 'ordering_' list is important!
        # Note that this implementation is O(n^2) theoretically, but
        # supposedly with very low constant factors.
        processed = np.zeros(X.shape[0], dtype=bool)
        ordering = np.zeros(X.shape[0], dtype=int)
        for ordering_idx in range(X.shape[0]):
            # Choose next based on smallest reachability distance
            # (And prefer smaller ids on ties, possibly np.inf!)
            index = np.where(processed == 0)[0]
            point = index[np.argmin(self.reachability_[index])]

            processed[point] = True
            ordering[ordering_idx] = point
            if self.core_distances_[point] != np.inf:
                self._set_reach_dist(point, processed, X, nbrs)
        return ordering

    def _set_reach_dist(self, point_index, processed, X, nbrs):
        P = X[point_index:point_index + 1]
        # Assume that radius_neighbors is faster without distances
        # and we don't need all distances, nevertheless, this means
        # we may be doing some work twice.
        indices = nbrs.radius_neighbors(P, radius=self.max_eps,
                                        return_distance=False)[0]

        # Getting indices of neighbors that have not been processed
        unproc = np.compress((~np.take(processed, indices)).ravel(),
                             indices, axis=0)
        # Neighbors of current point are already processed.
        if not unproc.size:
            return

        # Only compute distances to unprocessed neighbors:
        if self.metric == 'precomputed':
            dists = X[point_index, unproc]
        else:
            dists = pairwise_distances(P, np.take(X, unproc, axis=0),
                                       self.metric, n_jobs=None).ravel()

        rdists = np.maximum(dists, self.core_distances_[point_index])
        improved = np.where(rdists < np.take(self.reachability_, unproc))
        self.reachability_[unproc[improved]] = rdists[improved]
        self.predecessor_[unproc[improved]] = point_index

    def extract_dbscan(self, eps):

        check_is_fitted(self, 'reachability_')

        if eps > self.max_eps:
            raise ValueError('Specify an epsilon smaller than %s. Got %s.'
                             % (self.max_eps, eps))

        if eps * 5.0 > (self.max_eps * 1.05):
            warnings.warn(
                "Warning, max_eps (%s) is close to eps (%s): "
                "Output may be unstable." % (self.max_eps, eps),
                RuntimeWarning, stacklevel=2)
        # Stability warning is documented in _extract_dbscan method...

        return _extract_dbscan(self.ordering_, self.core_distances_,
                               self.reachability_, eps)


def _extract_dbscan(ordering, core_distances, reachability, eps):
    n_samples = len(core_distances)
    is_core = np.zeros(n_samples, dtype=bool)
    labels = np.zeros(n_samples, dtype=int)

    far_reach = reachability > eps
    near_core = core_distances <= eps
    labels[ordering] = np.cumsum(far_reach[ordering] & near_core[ordering]) - 1
    labels[far_reach & ~near_core] = -1
    is_core[near_core] = True
    return np.arange(n_samples)[is_core], labels


def _extract_optics(ordering, reachability, maxima_ratio=.75,
                    rejection_ratio=.7, similarity_threshold=0.4,
                    significant_min=.003, min_cluster_size=.005,
                    min_maxima_ratio=0.001):
    # Extraction wrapper
    # according to Ankerst M. et.al. 1999 (p. 5), for a small enough
    # generative distance epsilong, there should be more than one INF.
    if np.all(np.isinf(reachability)):
        raise ValueError("All reachability values are inf. Set a larger"
                         " max_eps.")
    normalization_factor = np.max(reachability[reachability < np.inf])
    reachability = reachability / normalization_factor
    reachability_plot = reachability[ordering].tolist()
    root_node = _automatic_cluster(reachability_plot, ordering,
                                   maxima_ratio, rejection_ratio,
                                   similarity_threshold, significant_min,
                                   min_cluster_size, min_maxima_ratio)
    leaves = _get_leaves(root_node, [])
    # Start cluster id's at 0
    clustid = 0
    n_samples = len(reachability)
    is_core = np.zeros(n_samples, dtype=bool)
    labels = np.full(n_samples, -1, dtype=int)
    # Start all points as non-core noise
    for leaf in leaves:
        index = ordering[leaf.start:leaf.end]
        labels[index] = clustid
        is_core[index] = 1
        clustid += 1
    return np.arange(n_samples)[is_core], labels


def _automatic_cluster(reachability_plot, ordering,
                       maxima_ratio, rejection_ratio,
                       similarity_threshold, significant_min,
                       min_cluster_size, min_maxima_ratio):
    min_neighborhood_size = 2
    if min_cluster_size <= 1:
        min_cluster_size = max(2, min_cluster_size * len(ordering))
    neighborhood_size = int(min_maxima_ratio * len(ordering))

    # Again, should this check < min_samples, should the parameter be public?
    if neighborhood_size < min_neighborhood_size:
        neighborhood_size = min_neighborhood_size

    local_maxima_points = _find_local_maxima(reachability_plot,
                                             neighborhood_size)
    root_node = _TreeNode(ordering, 0, len(ordering), None)
    _cluster_tree(root_node, None, local_maxima_points,
                  reachability_plot, ordering, min_cluster_size,
                  maxima_ratio, rejection_ratio,
                  similarity_threshold, significant_min)

    return root_node


class _TreeNode(object):
    # automatic cluster helper classes and functions
    def __init__(self, points, start, end, parent_node):
        self.points = points
        self.start = start
        self.end = end
        self.parent_node = parent_node
        self.children = []
        self.split_point = -1


def _is_local_maxima(index, reachability_plot, neighborhood_size):
    right_idx = slice(index + 1, index + neighborhood_size + 1)
    left_idx = slice(max(1, index - neighborhood_size - 1), index)
    return (np.all(reachability_plot[index] >= reachability_plot[left_idx]) and
            np.all(reachability_plot[index] >= reachability_plot[right_idx]))


def _find_local_maxima(reachability_plot, neighborhood_size):
    local_maxima_points = {}
    # 1st and last points on Reachability Plot are not taken
    # as local maxima points
    for i in range(1, len(reachability_plot) - 1):
        # if the point is a local maxima on the reachability plot with
        # regard to neighborhood_size, insert it into priority queue and
        # maxima list
        if (reachability_plot[i] > reachability_plot[i - 1] and
                reachability_plot[i] >= reachability_plot[i + 1] and
                _is_local_maxima(i, np.array(reachability_plot),
                                 neighborhood_size) == 1):
            local_maxima_points[i] = reachability_plot[i]

    return sorted(local_maxima_points,
                  key=local_maxima_points.__getitem__, reverse=True)


def _cluster_tree(node, parent_node, local_maxima_points,
                  reachability_plot, reachability_ordering,
                  min_cluster_size, maxima_ratio, rejection_ratio,
                  similarity_threshold, significant_min):
    if len(local_maxima_points) == 0:
        return  # parent_node is a leaf

    # take largest local maximum as possible separation between clusters
    s = local_maxima_points[0]
    node.split_point = s
    local_maxima_points = local_maxima_points[1:]

    # create two new nodes and add to list of nodes
    node_1 = _TreeNode(reachability_ordering[node.start:s],
                       node.start, s, node)
    node_2 = _TreeNode(reachability_ordering[s + 1:node.end],
                       s + 1, node.end, node)
    local_max_1 = []
    local_max_2 = []

    for i in local_maxima_points:
        if i < s:
            local_max_1.append(i)
        if i > s:
            local_max_2.append(i)

    node_list = []
    node_list.append((node_1, local_max_1))
    node_list.append((node_2, local_max_2))

    if reachability_plot[s] < significant_min:
        node.split_point = -1
        # if split_point is not significant, ignore this split and continue
        return

    # only check a certain ratio of points in the child
    # nodes formed to the left and right of the maxima
    # ...should check_ratio be a user settable parameter?
    check_ratio = .8
    check_value_1 = int(np.round(check_ratio * len(node_1.points)))
    check_value_2 = int(np.round(check_ratio * len(node_2.points)))
    avg_reach1 = np.mean(reachability_plot[(node_1.end -
                                            check_value_1):node_1.end])
    avg_reach2 = np.mean(reachability_plot[node_2.start:(node_2.start
                                                         + check_value_2)])

    if ((avg_reach1 / maxima_ratio) > reachability_plot[s] or
            (avg_reach2 / maxima_ratio) > reachability_plot[s]):

        if (avg_reach1 / rejection_ratio) < reachability_plot[s]:
            # reject node 2
            node_list.remove((node_2, local_max_2))
        if (avg_reach2 / rejection_ratio) < reachability_plot[s]:
            # reject node 1
            node_list.remove((node_1, local_max_1))
        if ((avg_reach1 / rejection_ratio) >= reachability_plot[s] and
                (avg_reach2 / rejection_ratio) >= reachability_plot[s]):
            # since split_point is not significant,
            # ignore this split and continue (reject both child nodes)
            node.split_point = -1
            _cluster_tree(node, parent_node, local_maxima_points,
                          reachability_plot, reachability_ordering,
                          min_cluster_size, maxima_ratio, rejection_ratio,
                          similarity_threshold, significant_min)
            return

    # remove clusters that are too small
    if (len(node_1.points) < min_cluster_size and
            node_list.count((node_1, local_max_1)) > 0):
        # cluster 1 is too small
        node_list.remove((node_1, local_max_1))
    if (len(node_2.points) < min_cluster_size and
            node_list.count((node_2, local_max_2)) > 0):
        # cluster 2 is too small
        node_list.remove((node_2, local_max_2))
    if not node_list:
        # parent_node will be a leaf
        node.split_point = -1
        return

    # Check if nodes can be moved up one level - the new cluster created
    # is too "similar" to its parent, given the similarity threshold.
    bypass_node = 0
    if parent_node is not None:
        if ((node.end - node.start) / (parent_node.end - parent_node.start) >
                similarity_threshold):
            parent_node.children.remove(node)
            bypass_node = 1

    for nl in node_list:
        if bypass_node == 1:
            parent_node.children.append(nl[0])
            _cluster_tree(nl[0], parent_node, nl[1],
                          reachability_plot, reachability_ordering,
                          min_cluster_size, maxima_ratio, rejection_ratio,
                          similarity_threshold, significant_min)
        else:
            node.children.append(nl[0])
            _cluster_tree(nl[0], node, nl[1], reachability_plot,
                          reachability_ordering, min_cluster_size,
                          maxima_ratio, rejection_ratio,
                          similarity_threshold, significant_min)


def _get_leaves(node, arr):
    if node is not None:
        if node.split_point == -1:
            arr.append(node)
        for n in node.children:
            _get_leaves(n, arr)
    return arr
