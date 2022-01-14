# Necessary packages
import os
import numpy as np
from sklearn.metrics import roc_auc_score


def load_data():
    path = './results/twins'
    path_list = os.listdir(path)
    print(path_list)
    for name in path_list:
        if not name.startswith('results_'):
            continue
        print('process {}'.format(name))

        # load pred in valid and test data
        valid_path = os.path.join(path, name, 'result.npz')
        test_path = os.path.join(path, name, 'result.test.npz')
        valid_res = np.load(valid_path)
        predictions = valid_res['pred']
        valid_index = valid_res['val'][0]
        # load data
        data = np.load(
            './data/twins/twins_data.train.npz')
        y = np.concatenate((data['yf'][valid_index], data['ycf'][valid_index]), axis=1)
        n_units, _, n_exp, n_outputs = predictions.shape
        roc_auc_list = []
        fact_auc_list = []
        for i_out in range(n_outputs):
            y_hat = predictions[valid_index, :, 0, i_out]
            auc, fact_auc = roc_auc(y, y_hat)
            roc_auc_list.append(auc)
            fact_auc_list.append(fact_auc)

        print('valid:')

        print('auc: {}'.format(np.mean(roc_auc_list)))

        print('test:')

        data = np.load(
            './data/twins/twins_data.test.npz')
        y = np.concatenate((data['yf'], data['ycf']), axis=1)
        test_res = np.load(test_path)
        predictions = test_res['pred']
        n_units, _, n_exp, n_outputs = predictions.shape
        roc_auc_list = []
        fact_auc_list = []
        for i_out in range(n_outputs):
            y_hat = predictions[:, :, 0, i_out]
            auc, fact_auc = roc_auc(y, y_hat)
            roc_auc_list.append(auc)
            fact_auc_list.append(fact_auc)
        print('auc: {}'.format(np.mean(roc_auc_list)))


def roc_auc(y, y_hat):
    y_label = np.concatenate((y[:, 0], y[:, 1]), axis=0)
    y_label_pred = np.concatenate((y_hat[:, 0], y_hat[:, 1]), axis=0)
    roc_auc_val = roc_auc_score(y_label, y_label_pred)

    fact_roc_auc = roc_auc_score(y[:, 0], y_hat[:, 0])
    return roc_auc_val, fact_roc_auc


if __name__ == '__main__':
    load_data()
