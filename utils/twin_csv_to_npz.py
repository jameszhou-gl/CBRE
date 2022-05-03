from numpy import genfromtxt

# Necessary packages
import os
import numpy as np
from scipy.special import expit


def data_trans_twin(train_rate=0.8):
    """Trans twins data.

    Args:
      - train_rate: the ratio of training data

    Returns:
      - train_x: features in training data
      - train_t: treatments in training data
      - train_y: observed outcomes in training data
      - train_potential_y: potential outcomes in training data
      - test_x: features in testing data
      - test_potential_y: potential outcomes in testing data
    """

    # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
    ori_data = np.loadtxt("../data/twins/Twin_data.csv", delimiter=",", skiprows=1)
    train_npz = '../data/twins/twins_data.train.npz'
    test_npz = '../data/twins/twins_data.test.npz'

    # Define features
    x = ori_data[:, :30]
    no, dim = x.shape

    # Define potential outcomes
    potential_y = ori_data[:, 30:]
    # Die within 1 year = 1, otherwise = 0
    potential_y = np.array(potential_y < 9999, dtype=float)

    ## Assign treatment
    coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
    prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

    prob_t = prob_temp / (2 * np.mean(prob_temp))
    prob_t[prob_t > 1] = 1

    t = np.random.binomial(1, prob_t, [no, 1])
    t = t.reshape([no, ])

    ## Define observable outcomes
    yf = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
    yf = np.reshape(np.transpose(yf), [no, ])
    ycf = np.transpose(t) * potential_y[:, 0] + np.transpose(1 - t) * potential_y[:, 1]
    ycf = np.reshape(np.transpose(ycf), [no, ])

    ## Train/test division
    idx = np.random.permutation(no)
    train_idx = idx[:int(train_rate * no)]
    train_len = len(train_idx)
    test_idx = idx[int(train_rate * no):]
    test_len = len(test_idx)

    train_x = x[train_idx, :]
    train_x = np.reshape(train_x, [train_len, dim, 1])
    train_t = np.reshape(t[train_idx], [train_len, 1])
    train_yf = np.reshape(yf[train_idx], [train_len, 1])
    train_ycf = np.reshape(ycf[train_idx], [train_len, 1])

    test_x = x[test_idx, :]
    test_x = np.reshape(test_x, [test_len, dim, 1])
    test_t = np.reshape(t[test_idx], [test_len, 1])
    test_yf = np.reshape(yf[test_idx], [test_len, 1])
    test_ycf = np.reshape(ycf[test_idx], [test_len, 1])

    np.savez(train_npz, x=train_x, t=train_t, yf=train_yf, ycf=train_ycf, mu0=None, mu1=None)
    np.savez(test_npz, x=test_x, t=test_t, yf=test_yf, ycf=test_ycf, mu0=None, mu1=None)


data_trans_twin()
