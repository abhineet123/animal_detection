from collections import OrderedDict
from functools import partial
from time import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

import cluster_DSI_example

# Next line to silence pyflakes. This import is needed.
Axes3D


def run(x, y, color):
    # n_points = 1000
    # X, color = datasets.make_s_curve(n_points, random_state=0)

    n_points = x.shape[0]

    n_neighbors = 10
    n_components = 3

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    # gs = gridspec.GridSpec(2, 2)
    fig.suptitle(
        "Manifold Learning with %i points, %i neighbors" % (n_points, n_neighbors), fontsize=14
    )

    # Add 3d scatter plot
    # ax = fig.add_subplot(251, projection="3d")
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    # ax.view_init(4, -72)

    # Set-up manifold methods
    methods = OrderedDict()

    # LLE = partial(
    #     manifold.LocallyLinearEmbedding,
    #     n_neighbors=n_neighbors,
    #     n_components=n_components,
    #     eigen_solver="auto",
    # )
    # methods["LLE"] = LLE(method="standard")
    # methods["LTSA"] = LLE(method="ltsa")
    # methods["Hessian LLE"] = LLE(method="hessian")
    # methods["Modified LLE"] = LLE(method="modified")

    methods["Isometric Mapping"] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
    methods["Multidimensional Scaling"] = manifold.MDS(n_components, max_iter=100, n_init=1)
    methods["Spectral Embedding"] = manifold.SpectralEmbedding(
        n_components=n_components, n_neighbors=n_neighbors)
    methods["t-distributed Stochastic Neighbor Embedding"] = manifold.TSNE(n_components=n_components, init="pca",
                                                                           random_state=0)

    # Plot results
    for i, (label, method) in enumerate(methods.items()):
        t0 = time()
        x_reduced = method.fit_transform(x)
        t1 = time()
        print("%s: %.2g sec" % (label, t1 - t0))
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(x_reduced[:, 0], x_reduced[:, 1], x_reduced[:, 2], c=color,
                   )
        ax.view_init(4, -72)

        ax.set_title(label, fontsize=26)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        # ax.axis("tight")

        out_fname = label.lower().replace(' ', '_') + '.txt'

        cluster_DSI_example.run(x_reduced, y, out_fname)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    print()
