from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def k_means_elbow_method(data, attributes, save_plot_path=None, save_plot=False, xlabel='x', ylabel='y'):
    # biramo podskup podataka, koji zelimo iskoristiti za klasterovanje
    cluster_data = data[attributes]

    # biramo raspon klastera za koji zelimo proveravati kvalitet i koristimo metod K-sredina za svaku od opcija za K
    ec_cluster_options = range(1, 10)
    kmeans_res = [KMeans(n_clusters=i, init='k-means++')
                  for i in ec_cluster_options]

    # wcss - within-cluster sum of square
    wcss = [kmeans_res[i].fit(cluster_data).inertia_
            for i in range(len(kmeans_res))]

    f_ec = plt.figure(figsize=(6, 4))
    plt.plot(range(1, 10), wcss)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    if save_plot:
        f_ec.savefig(save_plot_path, bbox_inches='tight')


def k_means_clustering(data, n_clusters, attributes, save_plot_path=None, save_plot=False, show_cluster_means=True,
                       xlabel='x', ylabel='y'):

    # biramo podskup podataka za klasterovanje
    cluster_data = data[attributes]

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    y_kmeans = kmeans.fit_predict(cluster_data)

    centroids = kmeans.cluster_centers_

    # izracunate klastere dodajemo u tabelu kao atribut 'Cluster'
    cluster_data['Cluster'] = y_kmeans
    # srednje vrednosti za oba atributa za svaki klaster
    kmeans_mean_cluster = pd.DataFrame(round(cluster_data.groupby('Cluster').mean(), 1))

    colors = ['red', 'green', 'blue', 'yellow', 'pink', 'black', 'orange', 'purple', 'brown']

    colors = colors[:n_clusters]

    f_clusters = plt.figure(figsize=(6, 4))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(cluster_data[attributes[0]], cluster_data[attributes[1]],
                c=kmeans.labels_.astype(float), alpha=0.5)

    plt.scatter(centroids[:, 0], centroids[:, 1], c=colors)
    plt.show()

    if show_cluster_means:
        print(kmeans_mean_cluster)

    if save_plot:
        f_clusters.savefig(save_plot_path, bbox_inches='tight')


def hierarchy_dendrogram(data, attributes, max_d=None, save_plot_path=None, save_plot=False, show_cluster_means=True,
                         xlabel='x', ylabel='y'):

    from scipy.cluster.hierarchy import dendrogram, linkage

    cluster_data = data[attributes]

    f_dendro = plt.figure(figsize=(15, 10))
    # izracunamo rastojanja izmedju elemenata i kreiramo dendrogram
    Z = linkage(cluster_data, method='average')
    dendrogram = dendrogram(Z, truncate_mode='lastp', p=20, leaf_rotation=90, leaf_font_size=21, show_contracted=True)

    # vrednost na y-osi na kojoj cemo vrsiti odsecanje

    if max_d is not None:
        plt.axhline(y=max_d, c='k')
    plt.xlabel(xlabel, fontsize=21)
    plt.ylabel(ylabel, fontsize=21)
    plt.show()

    if save_plot:
        f_dendro.savefig(save_plot_path, bbox_inches='tight')

    return Z


def hierarchy_clustering(data, Z, n_clusters, attributes, max_d=None, save_plot_path=None, save_plot=False,
                         show_cluster_means=True, xlabel='x', ylabel='y'):
    from scipy.cluster.hierarchy import linkage, fcluster

    cluster_data = data[attributes]

    clusters = fcluster(Z, max_d, criterion='distance')
    # oduzimamo 1 da bi brojanje klastera krenulo od 0 ovo radimo samo da bismo mogli da trazimo ispravne indekse iz
    # liste sa bojama
    cluster_data['Cluster'] = clusters - 1

    colors = ['red', 'green', 'blue', 'yellow', 'pink', 'black', 'orange', 'purple', 'brown']

    colors = colors[:n_clusters]

    cluster_means = pd.DataFrame(round(cluster_data.groupby('Cluster').mean(), 1))
    centroids = np.array(cluster_means)
    cluster_means['CentroidColor'] = pd.DataFrame(colors)

    f_clusters = plt.figure(figsize=(6, 4))
    plt.scatter(cluster_data[attributes[0]], cluster_data[attributes[1]], c=clusters, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    if show_cluster_means:
        # stampamo srednje vrednosti klastera i boje centroida
        print(cluster_means)

    if save_plot:
        f_clusters.savefig(save_plot_path, bbox_inches='tight')


def dbscan_clustering(cluster_data):
    # koriscen kod sa primera sa sajta sklearn:
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py

    db = DBSCAN(eps=0.1, min_samples=200).fit(cluster_data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Procenjen broj klastera: %d' % n_clusters_)
    print('Procenjen broj tacaka koje predstavljaju sum: %d' % n_noise_)
    print("Silhouette koeficijent: %0.3f" % metrics.silhouette_score(cluster_data, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Crna boja za sum
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = np.array(cluster_data[class_member_mask & core_samples_mask])
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)

        xy = np.array(cluster_data[class_member_mask & ~core_samples_mask])
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

    plt.show()
