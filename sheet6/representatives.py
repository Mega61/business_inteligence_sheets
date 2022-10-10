import numpy as np
from scipy.stats import multivariate_normal as N
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def kMeans(D, k, eps=0.01, mu=None, max_iter=20):
    d = D.shape[1]
    if mu is None:
        mu = np.random.rand(k, d)*(np.max(D, axis=0) -
                                   np.min(D, axis=0))+np.min(D, axis=0)
    if type(mu) == list:
        mu = np.array(mu)
    s = False
    g = 0
    while not s:
        C = [np.argmin([np.linalg.norm(mu[j]-x)
                       for j in range(k)if not any(np.isnan(mu[j]))])for x in D]
        n = np.zeros(mu.shape)
        s = True
        for i in range(k):
            J = [D[j]for j in range(len(D))if C[j] == i]
            if len(J) > 0:
                n[i] = np.mean(J, axis=0)
            else:
                n[i] = np.random.rand(
                    1, d)*(np.max(D, axis=0)-np.min(D, axis=0))+np.min(D, axis=0)
            if np.linalg.norm(mu[i]-n[i]) > eps:
                s = False
        mu = n
        g += 1
        if g >= max_iter:
            s = True
    return C, mu


def EM(D, k, i, mu=None, cov=None, independent=False, max_iter=10):
    d = D.shape[1]
    if mu is None:
        mu = D[np.random.choice(range(len(D)), k)]
    if type(mu) != np.array:
        mu = np.array(mu)
    if cov is None:
        C = np.cov(D, rowvar=False)
        cov = [C for i in range(k)]
    P = [1/k for i in range(k)]
    s = False
    Q = 1
    while not s:
        Q += 1
        B = np.array(
            [list(N.pdf(D, mu[a], cov[a], allow_singular=True)*P[a])for a in range(k)])
        Z = np.sum(B, axis=0)
        T = Z != 0
        w = B[:, T]/Z[T]
        if w.shape[1] == 0:
            raise Exception(
                f"In iteration {iteration}, none of the points in the database has a measurable density anymore.")
        M = np.zeros(mu.shape)
        for i in range(k):
            wi = sum(w[i])
            M[i] = (sum([w[i, j]*x for j, x in enumerate(D[T])])/wi).reshape(1, d)
            if independent:
                cov[i] = sum([w[i, j]*np.diag([(x-mu[i])[u]**2 for u in range(d)])
                             for j, x in enumerate(D[T])])/wi
            else:
                cov[i] = sum([w[i, j]*np.matmul((x-mu[i]).reshape(d, 1),
                             (x-mu[i]).reshape(1, d))for j, x in enumerate(D[T])])/wi
            P[i] = wi/len(D[T])
        if np.abs(sum(P)-1) > 0.001:
            raise Exception("P vector "+str(P) +
                            " does not sum up to "+str(sum(P))+" != 1!")
        j = sum([np.linalg.norm(mu[z]-D[z])**2 for z in range(k)])
        s = j <= i or Q >= max_iter
        mu = M
    return np.round(w, 4), mu, cov, P


def getClusterings(D, k):
    i = 1
    C1 = kMeans(D, k, i)[0]
    Y = EM(D, k, i, independent=True)[0]
    r = EM(D, k, i, independent=False)[0]
    C2 = [np.argmax(i)for i in Y.T]
    C3 = [np.argmax(i)for i in r.T]
    return C1, C2, C3


def plotClusters(D, C, dimX, dimY, dimZ=None, ax=None):
    h = type(D) == pd.DataFrame
    W = D.columns[dimX]if h else dimX
    X = D.columns[dimY]if h else dimY
    b = D.columns[dimZ]if h and not dimZ is None else dimZ
    if type(D) == pd.DataFrame:
        D = D.values
    x = np.unique(C)
    K = not dimZ is None
    if ax is None:
        if K:
            I = plt.figure()
            ax = I.add_subplot(111, projection='3d')
        else:
            I, ax = plt.subplots()
    for ci in x:
        f = np.where(C == ci)[0]
        if K:
            ax.scatter(D[f, dimX], D[f, dimY], D[f, dimZ])
        else:
            ax.scatter(D[f, dimX], D[f, dimY])
    ax.set_xlabel(W)
    ax.set_ylabel(X)
    if K:
        ax.set_zlabel(b)
