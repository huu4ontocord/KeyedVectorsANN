# Copyright (C) 2017 Hiep Huu Nguyen <ontocord@gmail.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# See additional licenses at https://github.com/facebookresearch/pysparnn and https://radimrehurek.com/gensim/
# Based on Pysparnn and Gensim.
"""Extension of gensim's KeyedVectors using pysparnn's ANN indexer. Depends on gensim, numpy, sklearn and scipy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gensim, numpy, time, string
from sklearn.feature_extraction import DictVectorizer

from numpy.linalg import norm
from numpy import dot
import collections
import random
import numpy as np

if hasattr(string, 'maketrans'):
    trannum = string.maketrans("0123456789", "##########")
else:
    trannum = str.maketrans("0123456789", "##########")

## Modified from pysparnn. See https://github.com/facebookresearch/pysparnn for more details.

# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Defines a distance search structure"""

import abc
import numpy as np
import scipy.sparse
import scipy.spatial.distance

class MatrixMetricSearch(object):
    """A matrix representation out of features."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, features, records_data):
        """
        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            records_data: Data to return when a doc is matched. Index of
                corresponds to features.
        """
        self.matrix = features
        self.records_data = np.array(records_data, copy=False)

    def get_feature_matrix(self):
        return self.matrix

    def get_records(self):
        return self.records_data

    @staticmethod
    @abc.abstractmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return

    @staticmethod
    @abc.abstractmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return

    @abc.abstractmethod
    def _transform_value(self, val):
        """
        Args:
            val: A numeric value to be (potentially transformed).
        Returns:
            The transformed numeric value.
        """
        return

    @abc.abstractmethod
    def _distance(self, a_matrix):
        """
        Args:
            a_matrix: A matrix with rows that represent records
                to search against.
            records_data: Data to return when a doc is matched. Index of
                corresponds to features.
        Returns:
            A dense array representing distance.
        """
        return

    def nearest_search(self, features, k=1, max_distance=None):
        """Find the closest item(s) for each set of features in features_list.

        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            max_distance: Return items at most max_distance from the query
                point.

        Returns:
            For each element in features_list, return the k-nearest items
            and their distance scores
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """

        dist_matrix = self._distance(features)

        if max_distance is None:
            max_distance = float("inf")

        dist_filter = (dist_matrix <= max_distance)

        ret = []
        for i in range(dist_matrix.shape[0]):
            # these arrays are the length of the sqrt(index)
            # replacing the for loop by matrix ops could speed things up

            index = dist_filter[i]
            scores = dist_matrix[i][index]
            records = self.records_data[index]

            if scores.sum() < 0.0001 and len(scores) > 0:
                # they are all practically the same
                # we have to do this to prevent infinite recursion
                # TODO: would love an alternative solution, this is a critical loop
                lenScores = len(scores)
                arg_index = np.random.choice(lenScores, min(lenScores, k), replace=False)
            else:
                arg_index = np.argsort(scores)[:k]

            curr_ret = zip(scores[arg_index], records[arg_index])

            ret.append(curr_ret)

        return ret

class CosineDistance(MatrixMetricSearch):
    """A matrix that implements cosine distance search against it.

    cosine_distance = 1 - cosine_similarity

    Note: We want items that are more similar to be closer to zero so we are
    going to instead return 1 - cosine_similarity. We do this so similarity
    and distance metrics can be treated the same way.
    """

    def __init__(self, features, records_data):
        super(CosineDistance, self).__init__(features, records_data)

        m_c = self.matrix.copy()
        m_c.data **= 2
        self.matrix_root_sum_square = \
                np.sqrt(np.asarray(m_c.sum(axis=1)).reshape(-1))

    @staticmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return scipy.sparse.csr_matrix(features)

    @staticmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return scipy.sparse.vstack(matrix_list)

    def _transform_value(self, v):
        return v

    def _distance(self, a_matrix):
        """Vectorised cosine distance"""
        # what is the implmentation of transpose? can i change the order?
        dprod = self.matrix.dot(a_matrix.transpose()).transpose() * 1.0

        a_c = a_matrix.copy()
        a_c.data **= 2
        a_root_sum_square = np.asarray(a_c.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return 1 - dprod.multiply(magnitude).toarray()

class UnitCosineDistance(MatrixMetricSearch):
    """A matrix that implements cosine distance search against it.

    cosine_distance = 1 - cosine_similarity

    Note: We want items that are more similar to be closer to zero so we are
    going to instead return 1 - cosine_similarity. We do this so similarity
    and distance metrics can be treated the same way.

    Assumes unit-vectors and takes some shortucts:
      * Uses integers instead of floats
      * 1**2 == 1 so that operation can be skipped
    """

    def __init__(self, features, records_data):
        super(UnitCosineDistance, self).__init__(features, records_data)
        self.matrix_root_sum_square = \
                np.sqrt(np.asarray(self.matrix.sum(axis=1)).reshape(-1))

    @staticmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return scipy.sparse.csr_matrix(features)

    @staticmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return scipy.sparse.vstack(matrix_list)

    def _transform_value(self, v):
        return 1

    def _distance(self, a_matrix):
        """Vectorised cosine distance"""
        # what is the implmentation of transpose? can i change the order?
        dprod = self.matrix.dot(a_matrix.transpose()).transpose() * 1.0

        a_root_sum_square = np.asarray(a_matrix.sum(axis=1)).reshape(-1)
        a_root_sum_square = \
                a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)

        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return 1 - dprod.multiply(magnitude).toarray()

class SlowEuclideanDistance(MatrixMetricSearch):
    """A matrix that implements euclidean distance search against it.
    WARNING: This is not optimized.
    """

    def __init__(self, features, records_data):
        super(SlowEuclideanDistance, self).__init__(features, records_data)
        self.matrix = self.matrix

    @staticmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return np.array(features, ndmin=2, copy=False)

    @staticmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return np.vstack(matrix_list)

    def _transform_value(self, v):
        return v

    def _distance(self, a_matrix):
        """Euclidean distance"""

        return scipy.spatial.distance.cdist(a_matrix, self.matrix, 'euclidean')

class DenseCosineDistance(MatrixMetricSearch):
    """A matrix that implements cosine distance search against it.

    cosine_distance = 1 - cosine_similarity

    Note: We want items that are more similar to be closer to zero so we are
    going to instead return 1 - cosine_similarity. We do this so similarity
    and distance metrics can be treated the same way.
    """

    def __init__(self, features, records_data):
        super(DenseCosineDistance, self).__init__(features, records_data)

        self.matrix_root_sum_square = \
                np.sqrt((self.matrix**2).sum(axis=1).reshape(-1))

    @staticmethod
    def features_to_matrix(features):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return np.array(features, ndmin=2, copy=False)

    @staticmethod
    def vstack(matrix_list):
        """
        Args:
            val: A list of features to be formatted.
        Returns:
            The transformed matrix.
        """
        return np.vstack(matrix_list)

    def _transform_value(self, v):
        return v

    def _distance(self, a_matrix):
        """Vectorised cosine distance"""
        # what is the implmentation of transpose? can i change the order?
        dprod = self.matrix.dot(a_matrix.transpose()).transpose() * 1.0

        a_root_sum_square = (a_matrix**2).sum(axis=1).reshape(-1)
        a_root_sum_square = a_root_sum_square.reshape(len(a_root_sum_square), 1)
        a_root_sum_square = np.sqrt(a_root_sum_square)


        magnitude = 1.0 / (a_root_sum_square * self.matrix_root_sum_square)

        return 1 - (dprod * magnitude)

## Modified from pysparnn. See https://github.com/facebookresearch/pysparnn for more details.
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Defines a cluster pruing search structure to do K-NN Queries"""


def k_best(tuple_list, k):
    """For a list of tuples [(distance, value), ...] - Get the k-best tuples by
    distance.
    Args:
        tuple_list: List of tuples. (distance, value)
        k: Number of tuples to return.
    """
    tuple_lst = sorted(tuple_list, key=lambda x: x[0],
                       reverse=False)[:k]

    return tuple_lst

def filter_unique(tuple_list):
    """For a list of tuples [(distance, value), ...] - filter out duplicate
    values.
    Args:
        tuple_list: List of tuples. (distance, value)
    """

    added = set()
    ret = []
    for distance, value in tuple_list:
        if not value in added:
            ret.append((distance, value))
            added.add(value)
    return ret


def filter_distance(results, return_distance):
    """For a list of tuples [(distance, value), ...] - optionally filter out
    the distance elements.
    Args:
        tuple_list: List of tuples. (distance, value)
        return_distance: boolean to determine if distances should be returned.
    """
    if return_distance:
        return results
    else:
        return list([x for y, x in results])


class ClusterIndex(object):
    """Search structure which gives speedup at slight loss of recall.

       Uses cluster pruning structure as defined in:
       http://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html

       tldr - searching for a document in an index of K documents is naievely
           O(K). However you can create a tree structure where the first level
           is O(sqrt(K)) and each of the leaves are also O(sqrt(K)).

           You randomly pick sqrt(K) items to be in the top level. Then for
           the K doccuments you assign it to the closest neighbor in the top
           level.

           This breaks up one O(K) search into O(2 * sqrt(K)) searches which
           is much much faster when K is big.

           This generalizes to h levels. The runtime becomes:
               O(h * h_root(K))
    """

    def __init__(self, features,  names, records_data=None,
                 distance_type=CosineDistance,
                 matrix_size=None,
                 parent=None, k_clusters=10):
        """Create a search index composed of recursively defined
        matricies. Does recursive KNN search. See class docstring for a
        description of the method.

        Args:
            features: A csr_matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            records_data: Data to return when a doc is matched. Index of
                corresponds to records_features.
            distance_type: Class that defines the distance measure to use.
            matrix_size: Ideal size for matrix multiplication. This controls
                the depth of the tree. Defaults to 2 levels (approx). Highly
                reccomended that the default value is used.
        """

        self.is_terminal = False
        self.parent = parent
        self.distance_type = distance_type
        self.desired_matrix_size = matrix_size
        self.names = names
        self.k_clusters=k_clusters

        if records_data is None:
            records_data = list(range(len(names))) 
        features = distance_type.features_to_matrix(features)
        num_records = features.shape[0]

        if matrix_size is None:
            matrix_size = max(int(np.sqrt(num_records)), 1000)
        else:
            matrix_size = int(matrix_size)

        self.matrix_size = matrix_size

        num_levels = np.log(num_records)/np.log(self.matrix_size)

        if num_levels <= 1.4:
            self.is_terminal = True
            self.root = distance_type(features, records_data)
        else:
            self.is_terminal = False
            records_data =  np.array(records_data)

            records_index = list(np.arange(features.shape[0]))
            clusters_size = min(self.matrix_size, num_records)
            clusters_selection = random.sample(records_index, clusters_size)
            clusters_selection = features[clusters_selection]

            item_to_clusters = collections.defaultdict(list)

            root = distance_type(clusters_selection,
                                 list(np.arange(clusters_selection.shape[0])))

            rng_step = self.matrix_size
            for rng in range(0, features.shape[0], rng_step):
                max_rng = min(rng + rng_step, features.shape[0])
                records_rng = features[rng:max_rng]
                for i, clstrs in enumerate(root.nearest_search(records_rng, k=1)):
                    for _, cluster in clstrs:
                        item_to_clusters[cluster].append(i + rng)

            clusters = []
            cluster_keeps = []
            for k, clust_sel in enumerate(clusters_selection):
                clustr = item_to_clusters[k]
                if len(clustr) > 0:
                    index = ClusterIndex(
                                         self.distance_type.vstack(features[clustr]),
                                         self.names,
                                         records_data[clustr],
                                         distance_type=distance_type,
                                         matrix_size=self.matrix_size,
                                         parent=self)

                    clusters.append(index)
                    cluster_keeps.append(clust_sel)

            cluster_keeps = self.distance_type.vstack(cluster_keeps)
            clusters = np.array(clusters)

            self.root = distance_type(cluster_keeps, clusters)


    def insert(self, feature, record):
        """Insert a single record into the index.

        Args:
            feature: feature vector
            record: record to return as the result of a search
        """
        feature = self.distance_type.features_to_matrix(feature)
        nearest = self
        while not nearest.is_terminal:
            nearest = nearest.root.nearest_search(feature, k=1)
            _, nearest = nearest[0][0]

        cluster_index = nearest
        parent_index = cluster_index.parent
        while parent_index and cluster_index.matrix_size * 2 < \
                len(cluster_index.root.get_records()):
            cluster_index = parent_index
            parent_index = cluster_index.parent

        cluster_index._reindex(feature, record)



    def _get_child_data(self):
        """Get all of the features and corresponding records represented in the
        full tree structure.

        Returns:
            A tuple of (list(features), list(records)).
        """

        if self.is_terminal:
            return [self.root.get_feature_matrix()], [self.root.get_records()]
        else:
            result_features = []
            result_records = []

            for c in self.root.get_records():
                features, records = c._get_child_data()

                result_features.extend(features)
                result_records.extend(records)

            return result_features, result_records

    def _reindex(self, feature=None, record=None):
        """Rebuild the search index. Optionally add a record. This is used
        when inserting records to the index.

        Args:
            feature: feature vector
            record: record to return as the result of a search
        """

        features, records = self._get_child_data()

        flat_rec = []
        for x in records:
            flat_rec.extend(x)

        if feature != None and record != None:
            features.append(feature)
            flat_rec.append(record)

        self.__init__(self.distance_type.vstack(features), flat_rec, self.distance_type,
                self.desired_matrix_size, self.parent)


    def _search(self, features, k=1,
                max_distance=None, k_clusters=1):
        """Find the closest item(s) for each feature_list in.

        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            max_distance: Return items no more than max_distance away from the
                query point. Defaults to any distance.
            k_clusters: number of branches (clusters) to search at each level.
                This increases recall at the cost of some speed.

                Note: max_distance constraints are also applied.
                    This means there may be less than k_clusters searched at
                    each level.
                    This means each search will fully traverse at least one
                    (but at most k_clusters) clusters at each level.

        Returns:
            For each element in features_list, return the k-nearest items
            and their distance score
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]
        """
        if self.is_terminal:
            return self.root.nearest_search(features, k=k,
                                            max_distance=max_distance)
        else:
            ret = []
            nearest = self.root.nearest_search(features, k=k_clusters)

            for i, nearest_clusters in enumerate(nearest):
                curr_ret = []
                for distance, cluster in nearest_clusters:

                    cluster_items = cluster.\
                            search(features[i], k=k,
                                   k_clusters=k_clusters,
                                   max_distance=max_distance)

                    for elements in cluster_items:
                        elements = list(elements)
                        if len(elements) > 0:
                            curr_ret.extend(elements)
                ret.append(k_best(curr_ret, k))
            return ret


    def most_similar(self, mean, topn=1, k_clusters=None):
        if k_clusters==None:
            k_clusters = self.k_clusters
        names = self.names
        return [(names[int(w[1])],1.0-w[0]) for w in self.search(mean, k=topn, k_clusters=k_clusters)[0]]

    def search(self, features, k=1, max_distance=None, k_clusters=1,
            return_distance=True):
        """Find the closest item(s) for each feature_list in the index.

        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            max_distance: Return items no more than max_distance away from the
                query point. Defaults to any distance.
            k_clusters: number of branches (clusters) to search at each level.
                This increases recall at the cost of some speed.

                Note: max_distance constraints are also applied.
                    This means there may be less than k_clusters searched at
                    each level.

                    This means each search will fully traverse at least one
                    (but at most k_clusters) clusters at each level.

        Returns:
            For each element in features_list, return the k-nearest items
            and (optionally) their distance score
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]

            Note: if return_distance == False then the scores are omitted
            [[item1_1, ..., item1_k],
             [item2_1, ..., item2_k], ...]
        """

        # search no more than 1k records at once
        # helps keap the matrix multiplies small
        batch_size = 1000
        results = []
        rng_step = batch_size
        features = self.distance_type.features_to_matrix(features)
        for rng in range(0, features.shape[0], rng_step):
            max_rng = min(rng + rng_step, features.shape[0])
            records_rng = features[rng:max_rng]

            results.extend(self._search(features=records_rng,
                                        k=k,
                                        max_distance=max_distance,
                                        k_clusters=k_clusters))

        return [filter_distance(res, return_distance) for res in results]

    def _print_structure(self, tabs=''):
        """Pretty print the tree index structure's matrix sizes"""
        print(tabs + str(self.root.matrix.shape[0]))
        if not self.is_terminal:
            for index in self.root.records_data:
                index.print_structure(tabs + '  ')

    def _max_depth(self):
        """Yield the max depth of the tree index"""
        if not self.is_terminal:
            max_dep = 0
            for index in self.root.records_data:
                max_dep = max(max_dep, index._max_depth())
            return 1 + max_dep
        else:
            return 1

    def _matrix_sizes(self, ret=None):
        """Return all of the matrix sizes within the index"""
        if ret is None:
            ret = []
        ret.append(len(self.root.records_data))
        if not self.is_terminal:
            for index in self.root.records_data:
                ret.extend(index._matrix_sizes())
        return ret


class MultiClusterIndex(object):
    """Search structure which provides query speedup at the loss of recall.

       There are two components to this.

       = Cluster Indexes =
       Uses cluster pruning index structure as defined in:
       http://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html

       Refer to ClusterIndex documentation.

       = Multiple Indexes =
       The MultiClusterIndex creates multiple ClusterIndexes. This method
       gives better recall at the cost of allocating more memory. The
       ClusterIndexes are created by randomly picking representative clusters.
       The randomization tends to do a pretty good job but it is not perfect.
       Elements can be assigned to clusters that are far from an optimal match.
       Creating more Indexes (random cluster allocations) increases the chances
       of finding a good match.

       There are three perameters that impact recall. Will discuss them all
       here:
       1) MuitiClusterIndex(matrix_size)
           This impacts the tree structure (see cluster index documentation).
           Has a good default value. By increasing this value your index will
           behave increasingly like brute force search and you will loose query
           efficiency. If matrix_size is greater than your number of records
           you get brute force search.
       2) MuitiClusterIndex.search(k_clusters)
           Number of clusters to check when looking for records. This increases
           recall at the cost of query speed. Can be specified dynamically.
       3) MuitiClusterIndex(num_indexes)
           Number of indexes to generate. This increases recall at the cost of
           query speed. It also increases memory usage. It can only be
           specified at index construction time.

           Compared to (2) this argument gives better recall and has comparable
           speed. This statement assumes default (automatic) matrix_size is
           used.
            Scenario 1:

            (a) num_indexes=2, k_clusters=1
            (b) num_indexes=1, k_clusters=2

            (a) will have better recall but consume 2x the memory. (a) will be
            slightly slower than (b).

            Scenario 2:

            (a) num_indexes=2, k_clusters=1, matrix_size >> records
            (b) num_indexes=1, k_clusters=2, matrix_size >> records

            This means that each index does a brute force search. (a) and (b)
            will have the same recall. (a) will be 2x slower than (b). (a) will
            consume 2x the memory of (b).

            Scenario 1 will be much faster than Scenario 2 for large data.
            Scenario 2 will have better recall than Scenario 1.
    """

    def __init__(self, features,  names, records_data=None,
                 distance_type=CosineDistance,
                 matrix_size=None, num_indexes=2, k_clusters=10):
        """Create a search index composed of multtiple ClusterIndexes. See
        class docstring for a description of the method.

        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            records_data: Data to return when a doc is matched. Index of
                corresponds to records_features.
            distance_type: Class that defines the distance measure to use.
            matrix_size: Ideal size for matrix multiplication. This controls
                the depth of the tree. Defaults to 2 levels (approx). Highly
                reccomended that the default value is used.
            num_indexes: Number of ClusterIndexes to construct. Improves recall
                at the cost of memory.
        """


        self.indexes = []
        self.names = names
        self.k_clusters=k_clusters
        for _ in range(num_indexes):
            self.indexes.append((ClusterIndex(features, names, records_data, 
                                              distance_type, matrix_size)))

    def insert(self, feature, record):
        """Insert a single record into the index.

        Args:
            feature: feature vector
            record: record to return as the result of a search
        """
        for ind in self.indexes:
            ind.insert(feature, record)


    def most_similar(self, mean, topn=1, k_clusters=None):
        if k_clusters==None:
            k_clusters = self.k_clusters
        names = self.names
        return [(names[int(w[1])],1.0-w[0]) for w in self.search(mean, k=topn, k_clusters=k_clusters)[0]]

    def search(self, features, k=1, max_distance=None, k_clusters=1,
               return_distance=True, num_indexes=None):
        """Find the closest item(s) for each feature_list in the index.

        Args:
            features: A matrix with rows that represent records
                (corresponding to the elements in records_data) and columns
                that describe a point in space for each row.
            k: Return the k closest results.
            max_distance: Return items no more than max_distance away from the
                query point. Defaults to any distance.
            k_clusters: number of branches (clusters) to search at each level
                within each index. This increases recall at the cost of some
                speed.

                Note: max_distance constraints are also applied.
                    This means there may be less than k_clusters searched at
                    each level.

                    This means each search will fully traverse at least one
                    (but at most k_clusters) clusters at each level.
            num_indexes: number of indexes to search. This increases recall at
                the cost of some speed. Can not be larger than the number of
                num_indexes that was specified in the constructor. Defaults to
                searching all indexes.

        Returns:
            For each element in features_list, return the k-nearest items
            and (optionally) their distance score
            [[(score1_1, item1_1), ..., (score1_k, item1_k)],
             [(score2_1, item2_1), ..., (score2_k, item2_k)], ...]

            Note: if return_distance == False then the scores are omitted
            [[item1_1, ..., item1_k],
             [item2_1, ..., item2_k], ...]
        """
        results = []
        if num_indexes is None:
            num_indexes = len(self.indexes)
        for ind in self.indexes[:num_indexes]:
            results.append(ind.search(features, k, max_distance,
                                      k_clusters, True))
        ret = []
        for r in np.hstack(results):
            ret.append(
                filter_distance(
                    k_best(filter_unique(r), k),
                    return_distance
                )
            )

        return ret

## Modified from keyedvectors.py from Gensim. See https://radimrehurek.com/gensim/ for more details.
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Radim Rehurek <me@radimrehurek.com>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

class KeyedVectorsANN(gensim.models.KeyedVectors):
    def __init__(self, indexer=None):
        gensim.models.KeyedVectors.__init__(self)
        self.indexer=indexer

    def most_similar(self, positive=[], negative=[], topn=1, restrict_vocab=None, indexer=None):
        positive = [word.translate(trannum) for word in positive]
        negative = [word.translate(trannum) for word in negative]
        if topn==False:
            return gensim.models.KeyedVectors.most_similar(self, positive=positive, negative=negative, topn=topn, restrict_vocab=restrict_vocab, indexer=None)
        if self.indexer is not None and indexer is None:
            indexer = self.indexer

        return gensim.models.KeyedVectors.most_similar(self, positive=positive, negative=negative, topn=topn, restrict_vocab=restrict_vocab, indexer=indexer)

    def accuracy_indexer(self, questions, case_insensitive=True, most_similar=most_similar, indexer=None):
        sections, section = [], None
        original_vocab = self.vocab     
        for line_no, line in enumerate(gensim.utils.smart_open(questions)):
            # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
            line = gensim.utils.to_unicode(line)
            if line.startswith(': '):
                # a new section starts => store the old section
                if section:
                    sections.append(section)
                    self.log_accuracy(section)
                section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
            else:
                if not section:
                    raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
                try:
                    if case_insensitive:
                        a, b, c, expected = [word.lower() for word in line.split()]
                    else:
                        a, b, c, expected = [word for word in line.split()]
                except:
                    continue
                if a not in original_vocab or b not in original_vocab or c not in original_vocab or expected not in original_vocab :
                    continue

                ignore = set([a, b, c])  # input words to be ignored
                predicted = None
                # find the most likely prediction, ignoring OOV words and input words
                sims = most_similar(self, positive=[b, c], negative=[a], topn=5, indexer=indexer)
                #print (a, b, c, expected, sims)
                predicted = [w[0] for w in sims if w[0] not in ignore]
                if predicted:
                    if  expected in predicted: # in the top 5, exepcted == predicted[0]
                        section['correct'].append((a, b, c, expected))
                    else:
                        section['incorrect'].append((a, b, c, expected))
                else:
                        section['incorrect'].append((a, b, c, expected))
        if section:
            # store the last section, too
            sections.append(section)


        total = {
            'section': 'total',
            'correct': sum((s['correct'] for s in sections), []),
            'incorrect': sum((s['incorrect'] for s in sections), []),

        }
        sections.append(total)
        return sections

def prepareANNModel(googleFileIn, googleFileOut, createSynonyms=True):
    kv = KeyedVectorsANN.load_word2vec_format(googleFileIn, binary=True)
    #print ("Google vectors len="+str(len(kv.vocab.keys())))
    names1 = kv.index2word

    # for words with different cases, use the word with the highest
    # count, assuming index2word is ordered from greatest to lease.
    # also trim out words that can't be saved as ascii.  TODO -
    # collapse all websites into a protoypical website vector, and
    # similar with emails
    for w in names1:
        if kv.vocab.get(w) == None:
            continue
        w2 = w.lower()
        if "@" in w2 or ".co" in w2 or ".org" in w2 or ".gov" in w2 or ".edu" in w2 or "www" in w2 or "http:" in w2 or ".net" in w2:
            del kv.vocab[w]                    
            continue
        try:
            str.encode(w, "ascii")
        except:
            del kv.vocab[w]
            continue
        w2 = w.lower()
        if w2 != w:
            if kv.vocab.get(w2) != None and kv.vocab[w].index < kv.vocab[w2].index:
                print (w2+"<C"+w)
                kv.vocab[w2].index = kv.vocab[w].index

    vecs = []
    names = []
    collapse = []
    i = 0
    for w in names1:
        if kv.vocab.get(w) == None:
            continue
        w2 = w.lower()
        if "@" in w2 or ".co" in w2 or ".org" in w2 or ".gov" in w2 or ".edu" in w2 or "www" in w2 or "http:" in w2 or ".net" in w2:
            del kv.vocab[w]                    
            continue
        try:
            str.encode(w, "ascii")
        except:
            del kv.vocab[w]
            continue
        if len(w) <= 25 and w.islower() and (kv.vocab[w].index < 50000 or ("-" not in w and "_" not in w)):
            vecs.append(kv[w])
            names.append(w)
            kv.vocab[w].index = i
            i+= 1
            #print(w)
        else:
            if len(w) <= 25 and ("-" in w or  "_" in w):
                collapse.append((w, kv[w], kv.vocab[w].index))
            else:
                del kv.vocab[w]            

    kv.syn0 = np.array(vecs)
    kv.syn0norm = None
    kv.index2word = names
    # set the k_clusters to 10. change this parameter to a higher
    # number for higher precisison, but it will take longer to search!
    cluster_index = ClusterIndex(kv.syn0, names, distance_type=DenseCosineDistance, k_clusters=10) 
    kv.indexer = cluster_index

    if createSynonyms:
        # create a rough synonym structure
        for wVec in collapse:
            w = wVec[0]
            vec = wVec[1]
            idx = wVec[2]
            w2 = w.lower()
            if (not w.islower()) and kv.vocab.get(w2) != None:
                del kv.vocab[w]
                continue
            ret = cluster_index.most_similar(vec, k_clusters=1) # just do a fast but lower accuracy search
            score = ret[0][1]
            w3 = ret[0][0]
            if w3 == w2:
                ret = ret[1:]
            if score >= 0.3:
                print (ret, w2)
            # collapse compound words that are close to what is already in the vocab based on ranking
            collapse=((score >= 0.6 and idx < 250000) or (score >= 0.7 and idx < 500000) or (score >= 0.8 and idx < 1000000) or (score >= 0.9))
            if not collapse and ((score >= 0.5 and idx < 250000) or (score >= 0.6 and idx < 500000) or (score >= 0.7 and idx < 1000000) or (score >= 0.8)):
                if len(w2) > 5:
                    w2a = w2[:len(w2)-2]
                else:
                    w2a = w2
                if w2a in w3:
                    collapse = True
                else:
                    w3Arr = w3.split("_")
                    for w3a in w3Arr:
                        if len(w3a) > 3 and w3a in w2:
                            collapse = True
                            break
                        if len(w3a) > 5:
                            w3a = w3a[:len(w3a)-2]
                        if len(w3a) > 3 and w3a in w2:
                            collapse = True
                            break
            if collapse:
                if w2 != w:
                    del kv.vocab[w]
                kv.vocab[w2] = kv.vocab[w3]
                print (w3, "<-", w2)
            else:
                del kv.vocab[w]
    else:
        for wVec in collapse:
            w = wVec[0]
            del kv.vocab[w]

    kv.save(googleFileOut)
    return kv

