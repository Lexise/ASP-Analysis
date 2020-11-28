import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import block_diag
from scipy.cluster.hierarchy import linkage
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

from scipy.cluster import hierarchy
import numpy as np


def seriation(Z, N, cur_index):
    """Returns the order implied by a hierarchical tree (dendrogram).

       :param Z: A hierarchical tree (dendrogram).
       :param N: The number of points given to the clustering process.
       :param cur_index: The position in the tree for the recursive traversal.

       :return: The order implied by the hierarchical tree Z.
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="ward"):
    """Returns a sorted distance matrix.

       :param dist_mat: A distance matrix.
       :param method: A string in ["ward", "single", "average", "complete"].

        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    """
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method)
    res_order = seriation(res_linkage, N, N + N - 2)
    # seriated_dist = np.zeros((N, N))
    # a, b = np.triu_indices(N, k=1)
    # seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    # seriated_dist[b, a] = seriated_dist[a, b]
    # return seriated_dist, res_order, res_linkage
    return res_order


def innovative_correlation_clustering(correlation_matrix):


    set_list = pd.DataFrame({
        "name": [],
        "content": []
    })
    arg_list = correlation_matrix.index
    for arg in arg_list:
        correlation = correlation_matrix[arg]  # place to change
        non_zero_set = []
        for index, arg_set in set_list.iterrows():
            if any(correlation[list(arg_set.content)]) == False:
                continue
            else:
                non_zero_set.append(index)

        if len(non_zero_set) == 0:
            i = len(set_list) + 1
            set_list.loc[i - 1] = ['A' + str(i), [arg]]  # A_k+1
        elif len(non_zero_set) == 1:
            set_idx = non_zero_set[0]
            a = list(set_list[set_list.index == set_idx].content)[0]
            set_content = a + [arg]
            set_list.loc[[set_idx], 'content'] = [set_content]
        elif len(non_zero_set) > 1:
            selected = set_list.loc[non_zero_set, :]
            merge_selected = [arg]   #changed
            for i in selected.content.values:
                merge_selected += i
            set_list.loc[[non_zero_set[0]], "content"] = [merge_selected]
            set_list.drop(non_zero_set[1:], inplace=True)



    set_list["len_set"] = [len(a) for a in set_list.content]
    sorted_set_list = set_list.sort_values(by=['len_set'])
    all_new_order = []
    for idx, raw in sorted_set_list.iterrows():
        choosed_arg = raw.content
        cluster = correlation_matrix.loc[choosed_arg, choosed_arg]
        distances = np.sqrt((1 - cluster) / 2)
        res_order = compute_serial_matrix(distances.values, method='single')
        new_order = [cluster.index[i] for i in res_order]
        all_new_order.extend(new_order)

    return all_new_order


def abs_optimal_leaf_ordering(correlation_matrix):

    # D = pdist(correlation_matrix, 'euclidean')
    # Z = linkage(D, 'ward')
    # optimal_Z = optimal_leaf_ordering(Z, D)
    # N = len(correlation_matrix)
    # new_order = seriation(optimal_Z, N, N + N - 2)
    # new = [correlation_matrix.index[i] for i in new_order]


    X = correlation_matrix  # np.random.randn(10,10)
    Z = hierarchy.ward(X)
    hierarchy.leaves_list(Z)

    new_order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))
    new = [correlation_matrix.index[i] for i in new_order]
    return new

def my_optimal_leaf_ordering(correlation_matrix):
    set_list = pd.DataFrame({
        "name": [],
        "content": []
    })
    arg_list = correlation_matrix.index
    for arg in arg_list:
        correlation = correlation_matrix[arg]  # place to change
        non_zero_set = []
        for index, arg_set in set_list.iterrows():
            if any(correlation[list(arg_set.content)]) == False:
                continue
            else:
                non_zero_set.append(index)

        if len(non_zero_set) == 0:
            i = len(set_list) + 1
            set_list.loc[i - 1] = ['A' + str(i), [arg]]  # A_k+1
        elif len(non_zero_set) == 1:
            set_idx = non_zero_set[0]
            a = list(set_list[set_list.index == set_idx].content)[0]
            set_content = a + [arg]
            set_list.loc[[set_idx], 'content'] = [set_content]
        elif len(non_zero_set) > 1:
            selected = set_list.loc[non_zero_set, :]
            merge_selected = [arg]
            for i in selected.content.values:
                merge_selected += i
            set_list.loc[[non_zero_set[0]], "content"] = [merge_selected]
            set_list.drop(non_zero_set[1:], inplace=True)

    len_set = []
    for idx, raw in set_list.iterrows():
        len_set.append(len(raw.content))
    set_list["len_set"] = len_set
    sorted_set_list = set_list.sort_values(by=['len_set'])

    all_new_order = []
    for idx, raw in sorted_set_list.iterrows():
        choosed_arg = raw.content
        cluster = correlation_matrix.loc[choosed_arg, choosed_arg]
        new_order = abs_optimal_leaf_ordering(cluster)
        all_new_order.extend(new_order)
    return all_new_order
    #new_test = correlation_matrix.reindex(index=all_new_order, columns=all_new_order)


