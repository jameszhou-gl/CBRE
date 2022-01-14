import tensorflow as tf
import numpy as np

SQRT_CONST = 1e-10

FLAGS = tf.app.flags.FLAGS


def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp['x'].shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction * n)
        n_train = n - n_valid
        I = np.random.permutation(range(0, n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid


def log(logfile, st):
    """ Log a string in a file """
    with open(logfile, 'a') as f:
        f.write(st + '\n')
    print(st)


def save_config(fname):
    """ Save configuration """
    flagdict = FLAGS.__dict__['__flags']
    s = '\n'.join(['%s: %s' % (k, str(flagdict[k])) for k in sorted(flagdict.keys())])
    f = open(fname, 'w')
    f.write(s)
    f.close()


def preprocess(x, data_name='ihdp'):
    '''
    preprocess features in x
    '''
    from sklearn.preprocessing import StandardScaler
    if data_name == 'ihdp':
        cons_list = [0, 1, 2, 3, 4, 5]
        disc_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    elif data_name == 'jobs':
        cons_list = [0, 1, 6, 7, 8, 9, 10, 11, 12, 15]
        disc_list = [2, 3, 4, 5, 13, 14, 16]
    else:
        cons_list = []
        disc_list = []
        return x
    cons_x = x[:, cons_list]
    disc_x = x[:, disc_list]
    scaler = StandardScaler()
    trans_cons_x = scaler.fit_transform(cons_x)
    new_x = np.concatenate([trans_cons_x, disc_x], axis=1)
    return new_x


def load_data(fname):
    """ Load data set """
    data_in = np.load(fname)
    data = dict()
    data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
    try:
        data['ycf'] = data_in['ycf']
    except KeyError as e:
        data['ycf'] = None
    # data['x'] = preprocess(data['x'])
    data['HAVE_TRUTH'] = data['ycf'] is not None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data


def load_sparse(fname):
    """ Load sparse data set """
    E = np.loadtxt(open(fname, "rb"), delimiter=",")
    H = E[0, :]
    n = int(H[0])
    d = int(H[1])
    E = E[1:, :]
    S = sparse.coo_matrix((E[:, 2], (E[:, 0] - 1, E[:, 1] - 1)), shape=(n, d))
    S = S.todense()

    return S


def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return tf.sqrt(tf.clip_by_value(x, lbound, np.inf))


def lindisc(X, p, t):
    ''' Linear MMD '''

    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)

    mean_control = tf.reduce_mean(Xc, reduction_indices=0)
    mean_treated = tf.reduce_mean(Xt, reduction_indices=0)

    c = tf.square(2 * p - 1) * 0.25
    f = tf.sign(p - 0.5)

    mmd = tf.reduce_sum(tf.square(p * mean_treated - (1 - p) * mean_control))
    mmd = f * (p - 0.5) + safe_sqrt(c + mmd)

    return mmd


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keep_dims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


def pdist2(X, Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X, Y))


def pop_dist(X, t):
    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt, Xc)
    return M


def simplex_project(x, k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x, axis=0)[::-1]
    nu = (np.cumsum(mu) - k) / range(1, d + 1)
    I = [i for i in range(0, d) if mu[i] > nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x - theta, 0)
    return w
