import numpy as np

def explained_variance_view(x, z, w, intercept=None):
    M = len(w)
    N = z.shape[0]
    if intercept is None:
        intercept = [np.zeros(w[m].shape[0]) for m in range(M)]
    return np.array([1 - np.sum((x[m] - np.dot(z, w[m].T) - np.outer(np.ones(N), intercept[m]))**2 ) / np.sum((x[m]- np.mean(x[m], axis=0))**2) for m in range(M)])

def explained_variance_factor_view(x, z, w, intercept=None):
    K = z.shape[1]
    M = len(w)
    N = z.shape[0]
    if intercept is None:
        intercept = [np.zeros(w[m].shape[0]) for m in range(M)]
    return np.array([[1 - np.sum((x[m] - np.outer(z[:, k], w[m][:, k]) - np.outer(np.ones(N), intercept[m]))**2) / np.sum((x[m] - np.mean(x[m], axis=0))**2) for m in range(M)] for k in range(K)])

def explained_variance_factor(x, z, w, intercept=None):
    K = z.shape[1]
    M = len(w)
    N = z.shape[0]
    if intercept is None:
        intercept = [np.zeros(w[m].shape[0]) for m in range(M)]
    return np.array([1 - np.sum([
        np.sum((x[m] - np.outer(z[:,k], w[m][:, k]) - np.outer(np.ones(N), intercept[m]))**2) for m in range(M)]) / 
        np.sum([np.sum((x[m]- np.mean(x[m], axis=0))**2) for m in range(M)]) for k in range(K)])

def explained_variance_total(x, z, w, intercept=None):
    K = z.shape[1]
    M = len(w)
    N = z.shape[0]
    if intercept is None:
        intercept = [np.zeros(w[m].shape[0]) for m in range(M)]
    return np.array(1 - np.sum([
        np.sum((x[m] - np.dot(z, w[m].T) - np.outer(np.ones(N), intercept[m]))**2) for m in range(M)]) / 
        np.sum([np.sum((x[m]- np.mean(x[m], axis=0))**2) for m in range(M)]))