import numpy as np
import pandas as pd
from tensorflow import Graph, Session
from scipy.stats import entropy
from scipy.linalg import sqrtm
from numpy import trace
from keras import backend as K
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix


graph = Graph()
session = Session()


def rmse(a, b):
    """"
    Root mean squared error between two arrays
    """
    return np.sqrt(((a - b)**2).mean())


def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname, "rb"), delimiter=",")
    H = E[0, :]
    n = int(H[0])
    d = int(H[1])
    E = E[1:, :]
    S = sparse.coo_matrix((E[:, 2], (E[:, 0] - 1, E[:, 1] - 1)), shape=(n, d))
    K = S.tocsr()
    return K


def mixup(X_train, y_train, bs=10, alpha=0.1):
    x1 = X_train[:bs]
    x2 = X_train[bs:]

    y1 = y_train[:bs]
    y2 = y_train[bs:]

    lam = np.random.beta(alpha, alpha, 1)
    x = (lam * x1 + (1. - lam) * x2)
    y = (lam * y1 + (1. - lam) * y2)
utils
    return x, y


def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)


def batch_generator(data, batch_size, replace=False):
    """Generate batches of data.
    Given a list of numpy arrays, it iterates over the list and returns batches of the same size.
    """
    all_examples_indices = len(data[0])
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=replace)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr


def batch_generator_weighted(data, batch_size, weights, replace=False):
    """Generate batches of data.
    Given a list of numpy arrays, it iterates over the list and returns batches of the same size.
    """
    all_examples_indices = len(data[0])
    weights = np.array(weights)/np.sum(weights)
    #print(weights.shape, (all_examples_indices))
    #exit()
    while True:
        mini_batch_indices = np.random.choice(all_examples_indices, size=batch_size, replace=replace, p = weights)
        tbr = [k[mini_batch_indices] for k in data]
        yield tbr

def gau_js_samples(sample0, sample1):
    pm = np.mean(sample0, axis=0)
    pv = np.cov(sample0.T)
    pv = pv + np.random.random(size=pv.shape)*0.2
    qm = np.mean(sample1, axis=0)
    qv = np.cov(sample1.T)
    qv = qv + np.random.random(size=qv.shape) * 0.2
    return gau_js(pm, pv, qm, qv)


def bh_distance(sample0, sample1):
    pm = np.mean(sample0, axis=0)

    pv = np.cov(sample0.T)
    pv = pv + np.random.random(size = pv.shape)*0.2
    qm = np.mean(sample1, axis=0)
    qv = np.cov(sample1.T)
    qv = qv + np.random.random(size=qv.shape) * 0.2

    p = (pv + qv) / 2
    diff = np.array([(qm - pm)])

    distance = (1/8.0) * diff.dot(np.linalg.inv(p)).dot(diff.T) + \
               0.5 * np.log(np.linalg.det(p)/(np.linalg.det(pv) * np.linalg.det(qv)))
    return distance


def wasser(sample0, sample1):
    pm = np.mean(sample0, axis=0)
    pv = np.cov(sample0.T)
    qm = np.mean(sample1, axis=0)
    qv = np.cov(sample1.T)
    s1 = sqrtm(pv).dot(qv).dot(sqrtm(pv))
    s2 = (pv + qv) - 2 * sqrtm(s1)
    distance = np.linalg.norm(pm - qm) + trace(s2)
    return distance


def geu_kl(pm, pv, qm, qv):


    dpv = np.linalg.det(pv)
    dqv = np.linalg.det(qv)
    # Inverses of diagonal covariances pv, qv
    iqv = np.linalg.inv(qv)
    # Difference between means pm, qm
    diff = qm - pm
    # KL(p||q)
    kl = (0.5 *
           (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
            + (np.matrix.trace(iqv.T.dot(pv)))         # + tr(\Sigma_q^{-1} * \Sigma_p)
            + diff.dot(iqv).dot(diff) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)))
    return kl


def gau_js(pm, pv, qm, qv):
    """
    Jensen-Shannon divergence between two Gaussians.  Also computes JS
    divergence between a single Gaussian pm, pv and a set of Gaussians
    qm, qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    return 0.5 * (geu_kl(pm, pv, qm, qv) + geu_kl(qm, qv, pm, pv))


def dump_csv(filename, data, columns):
    df = pd.DataFrame(columns = columns, data=data)
    df.to_csv(filename)


def entropy1(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

