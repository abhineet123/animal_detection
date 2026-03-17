'''
This is an example of computing CVIs on datasets after clustering.
It includes codes to compute the distance-based separability index (DSI).

Datasets:           Optical recognition of handwritten digits dataset (can add more)
CVIs:               DSI (can add more)
Clustering Methods: KMeans
                    Spectral Clustering
                    BIRCH
                    GaussianMixture (EM)
                    (can add more)

Related paper:      An Internal Cluster Validity Index Using a Distance-based Separability Measure
                    International Conference on Tools with Artificial Intelligence (ICTAI), 2020
                    https://arxiv.org/abs/2009.01328

By:                 Shuyue Guan
                    https://shuyueg.github.io/
'''

import numpy as np
import scipy.spatial.distance as distance
import sklearn.datasets as skdata
from scipy.stats import ks_2samp
from sklearn import cluster, mixture
from sklearn.preprocessing import StandardScaler


# ==================================================
# CVIs
# ==================================================

# load or define CVIs ###################################
#####################  DSI ##################{
def dists(data, dist_func=distance.euclidean):  # compute ICD
    num = data.shape[0]
    data = data.reshape((num, -1))
    dist = []
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            dist.append(dist_func(data[i], data[j]))
    return np.array(dist)


def dist_btw(a, b, dist_func=distance.euclidean):  # compute BCD
    a = a.reshape((a.shape[0], -1))
    b = b.reshape((b.shape[0], -1))
    dist = []
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            dist.append(dist_func(a[i], b[j]))
    return np.array(dist)


def separability_index_ks_2samp(X, labels):  # KS test on ICD and BCD
    classes = np.unique(labels)
    ks_values = []
    p_values = []

    alpha_to_c = (
        (0.2, 1.073),
        (0.15, 1.138),
        (0.1, 1.224),
        (0.05, 1.358),
        (0.025, 1.48),
        (0.01, 1.628),
        (0.005, 1.731),
        (0.001, 1.949),
    )

    for c in classes:
        pos = X[np.squeeze(labels == c)]
        neg = X[np.squeeze(labels != c)]

        dist_pos = dists(pos)
        distbtw = dist_btw(pos, neg)

        n = dist_pos.size
        m = distbtw.size
        ks_value, p_value = ks_2samp(dist_pos, distbtw)  # KS test

        critical_val_base = np.sqrt((n + m) / (n * m))

        alpha_to_critical_va = []
        alpha_to_pass = []
        for alpha, c in alpha_to_c:
            critical_val = c * critical_val_base
            critical_val_pass = (ks_value > critical_val)

            alpha_to_critical_va.append((alpha, critical_val))
            alpha_to_pass.append((alpha, critical_val_pass))

        # ks_value, p_value = ks_2samp(dist_pos, dist_pos)  # KS test
        ks_values.append(ks_value)
        p_values.append(p_value)

    D_sum = np.mean(ks_values)
    p_value_SUM = np.mean(p_values)
    return D_sum, p_value_SUM


def run(x=None, y=None, out_fname=None):
    measures = [
        # ('name',function),
        ('DSI', separability_index_ks_2samp)
    ]

    params = {
        'quantile': .3,
        'eps': .3,
        'damping': .9,
        'preference': -200,
        'n_neighbors': 10,
        'n_clusters': 3,
        'min_samples': 20,
        'xi': 0.05,
        'min_cluster_size': 0.1
    }

    np.random.seed(0)

    if out_fname is None:
        out_fname = 'dsi_results.txt'

    if x is None:
        digits = skdata.load_digits(n_class=10, return_X_y=True)

        i_dataset, dataset, algo_params = ('digits', digits, {'n_clusters': 10})
        params.update(algo_params)

        x, y = dataset
        y = np.squeeze(y)

    # normalize dataset for easier parameter selection
    try:
        x = StandardScaler().fit_transform(x)
    except:
        x = StandardScaler(with_mean=False).fit_transform(x)

    # ==================================================
    # Clustering Methods
    # ==================================================

    # load or define clustering methods ###################################
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])

    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")

    birch = cluster.Birch(n_clusters=params['n_clusters'])

    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    # put in clustering method list ###################################
    clustering_algorithms = [
        # ('name', method),
        ('REAL', {}),  # use true labels
        # ('KMeans', two_means),
        # ('SpectralClustering', spectral),
        # ('Birch', birch),
        # ('GaussianMixture', gmm)
    ]

    #########################################################################
    for name, algorithm in clustering_algorithms:  # loop clustering methods

        if name == 'REAL':  # use true labels as prediction
            y_pred = y
        else:

            algorithm.fit(x)  # apply

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(x)

        # with open('results.txt', 'a') as fw:  # new line for the next clustering method
        #     fw.write('\n')

        #########################################################################
        for meas_name, method in measures:  # loop CVIs
            if meas_name == 'DSI':
                score, p_value_SUM = method(x, y_pred)

            with open(out_fname, 'a') as fw:  # record CVIs
                fw.write('{}\t{}\t{:e}\t{:e}\n'.format(name, meas_name, score, p_value_SUM))


if __name__ == '__main__':
    run()
