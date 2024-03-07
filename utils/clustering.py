import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.plotting import axis_matras, bot_bar_plot, lin_colors


def gen_graph(similarity_matrix, threshold=0.8):
    """
    Generates a graph from the similarity_matrix (square dataframe). Each sample is a node, similarity - edge weight.
    Edges with weight lower than the threshold are ignored.
    Only nodes with at least 1 edge with weight above the threshold will be present in the final graph
    :param similarity_matrix:
    :param threshold:
    :return:
    """
    G = nx.Graph()
    for col_n in similarity_matrix:
        col = similarity_matrix[col_n].drop(col_n)
        mtr = col[col > threshold]
        for row_n, val in list(mtr.to_dict().items()):
            G.add_edge(col_n, row_n, weight=round(val, 2))
    return G


def leiden_community(
    correlation_matrix, threshold=1.4, n_iterations=2, random_state=42, **kwargs
):
    """
    Generates a graph from a correlation_matrix with weighted edges (weight<threshold excluded).
    Then detects communities using leiden algorithm (https://github.com/vtraag/leidenalg)
    :param correlation_matrix:
    :param threshold:
    :param n_iterations:
    :param random_state:
    :param kwargs:
    :return:
    """
    import igraph as ig
    import leidenalg as la

    G = gen_graph(correlation_matrix, threshold)

    # TODO replace graph generation with pure igraph
    IG = ig.Graph()
    IG.add_vertices(list(G.nodes))
    IG.add_edges(list(G.edges))
    IG.es['weight'] = [G[x[0]][x[1]]['weight'] for x in G.edges]

    la_out = la.find_partition(
        IG,
        la.ModularityVertexPartition,
        weights='weight',
        n_iterations=n_iterations,
        seed=random_state,
        **kwargs
    )
    return pd.Series(la_out.membership, index=IG.vs['name']) + 1


def dense_clustering( data, threshold=0.4, name='Dense_subtypes', method='leiden', **kwargs):
    """
    Generates a graph from the table features(cols)*samples(rows).
    Then performs community detection using a selected method leiden|louvain
    :param method:
    :param data:
    :param threshold:
    :param name:
    :return:
    """
    if method == 'leiden':
        partition = leiden_community(data.T.corr() + 1, threshold + 1, **kwargs)
    else:
        raise Exception('Unknown method')

    return partition.rename(name)


def clustering_profile_metrics(
    data, threshold_mm=(0.3, 0.6), step=0.025, method='leiden'
):
    """
    Iterates threshold in threshold_mm area with step. Calculates cluster separation metrics on each threshold.
    Returns a pd.DataFrame with the metrics
    :param data:
    :param threshold_mm:
    :param step:
    :param method:
    :return:
    """
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )

    cluster_metrics = {}

    for tr in tqdm(
        np.round(np.arange(threshold_mm[0], threshold_mm[1], step), 3)
    ):
        clusters_comb = dense_clustering(data, threshold=tr, method=method)
        cluster_metrics[tr] = {
            'ch': calinski_harabasz_score(data.loc[clusters_comb.index], clusters_comb),
            'db': davies_bouldin_score(data.loc[clusters_comb.index], clusters_comb),
            'sc': silhouette_score(data.loc[clusters_comb.index], clusters_comb),
            'N': len(clusters_comb.unique()),
            'perc': clusters_comb,
        }

    return pd.DataFrame(cluster_metrics).T


def clustering_profile_metrics_plot(cluster_metrics, num_clusters_ylim_max):
    """
    Plots a dataframe from clustering_profile_metrics
    :param cluster_metrics:
    :param num_clusters_ylim_max:
    :return: axis array
    """
    # necessary for correct x axis sharing
    cluster_metrics.index = [str(x) for x in cluster_metrics.index]

    af = axis_matras([3, 3, 3, 1, 2], sharex=True)

    ax = cluster_metrics.db.plot(ax=next(af), label='Davies Bouldin', color='#E63D06')
    ax.legend()

    ax = cluster_metrics.ch.plot(
        ax=next(af), label='Calinski Harabasz', color='#E63D06'
    )
    ax.legend()

    ax = cluster_metrics.sc.plot(ax=next(af), label='Silhouette score', color='#E63D06')
    ax.legend()

    ax = cluster_metrics.N.plot(
        kind='line', ax=next(af), label='# clusters', color='#000000'
    )
    ax.set_ylim(0, num_clusters_ylim_max)
    ax.legend()

    # display percentage for 10 clusters max
    clusters_perc = pd.DataFrame(
        [x.value_counts() for x in cluster_metrics.perc], index=cluster_metrics.index
    ).iloc[:, :10]

    ax = bot_bar_plot(
        clusters_perc,
        ax=next(af),
        legend=False,
        offset=0.5,
        palette=lin_colors(pd.Series(clusters_perc.columns), cmap=matplotlib.cm.tab20c),
    )

    ax.set_xticks(ax.get_xticks() - 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    ax.set_ylabel('Cluster %')

    return af


def clustering_select_best_tr(
    data,
    n_clusters=4,
    threshold_mm=(0.3, 0.6),
    step=0.025,
    method='leiden',
    num_clusters_ylim_max=7,
    plot=True,
):
    """
    Selects the best threshold for n_clusters separation using dense_clustering with selected method
        from threshold_mm with a paticular step
    :param data: dataframe with processes (rows - samples, columns - signatures)
    :param n_clusters: desired number of clusters
    :param threshold_mm: range of thresholds
    :param step: step to go through range of thresholds
    :param method: clusterization method - 'leiden'|'louvain'
    :param num_clusters_ylim_max: set y_lim for plot with number of clusters
    :param plot: whether to plot all matrics
    :return: the threshold to get n_clusters
    """
    cl_scs = clustering_profile_metrics(
        data, threshold_mm=threshold_mm, step=step, method=method
    )

    if plot:
        clustering_profile_metrics_plot(cl_scs, num_clusters_ylim_max)
        plt.show()

    cl_scs_filtered = cl_scs[cl_scs.N == n_clusters]

    if not len(cl_scs_filtered):
        raise Exception('No partition with n_clusters = {}'.format(n_clusters))

    cl_scs_filtered.sc += 1 - cl_scs_filtered.sc.min()
    return (
        (cl_scs_filtered.ch / cl_scs_filtered.db / cl_scs_filtered.sc)
        .sort_values()
        .index[-1]
    )