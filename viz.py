# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# TODO: refactor this
# TODO: Setup infrastructure for visualizing predictions/models
from dat import test_ps
from dat import comb_ps as train_ps

import time
from itertools import product as lproduct

import numpy as np
import pandas as pd
from pandas import DataFrame as DF

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO: read about seaborn
import seaborn as sb

from clfrs import load_clfr

# For reproducability of the results
np.random.seed(42)

def build_df(X,y):
    feat_cols = [ 'feat'+str(i) for i in range(len(X[0])) ]
    df = DF(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    return df, feat_cols

def perform_PCA(df, feat_cols):
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]
    df['pca-three'] = pca_result[:,2]
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

def plot_PCA_2d(df, n_labels, locs):
    ax = plt.figure(figsize=(16,10))
    sb.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sb.color_palette("hls", n_labels),
        data=df.loc[locs,:],
        legend="full",
        alpha=0.3
    )
    return ax

def plot_PCA_3d(df, locs):
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df.loc[locs,:]["pca-one"],
        ys=df.loc[locs,:]["pca-two"],
        zs=df.loc[locs,:]["pca-three"],
        c=df.loc[locs,:]["y"],
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    return ax

def perform_TSNE(df):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df)
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

def plot_TSNE(df, n_labels, support=[]):
    ax = plt.figure(figsize=(16,10))
    support_indicator = [1 if i in support else 0 for i in range(len(df['y']))]
    df['is_support'] = support_indicator
    for b in range(2):
        sb.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="y",
            palette=sb.color_palette("hls", n_labels),
            style="is_support",
            data=df[df.is_support==b],
            legend="full",
            alpha=(0.3 if b==0 else 1)
        )
    return ax

# def boundary_aux(xs, v):
#     T = list(zip(*xs))
#     bounds = [(min(x), max(x)) for x in T]
#     def partition(lb,ub):
#         tot = (ub-lb)
#         return [lb + (1/v)*i for i in range(int(v*tot))]
#     return [partition(b[0],b[1]) for b in bounds]
#
# def model_boundary(xs, v, model):
#     aux = boundary_aux(xs, v)
#     ps = list(lproduct(*aux))
#     preds = model.predict(ps)
#     return ps, preds


# b = boundary_aux([[0,0],[1,2]], 3)
# print(b)
# print(list(lproduct(*b)))
# print(list(lproduct(*[[0,1],["a","b"]])))



df, feat_cols = build_df(train_ps.xs,train_ps.ys)
locs = np.random.permutation(df.shape[0])

# perform_PCA(df, feat_cols)
# ax = plot_PCA_2d(df, 6, locs)
# plt.xlim(-5,5)
# plt.ylim(-5,5)
# plt.show()

fname = "tuned-['rbf']-1648221429.1081467"
lclfr, meta = load_clfr(fname)

perform_TSNE(df)
# plot_TSNE(df, 6, lclfr.support_)
plot_TSNE(df, 6)
plt.show()
