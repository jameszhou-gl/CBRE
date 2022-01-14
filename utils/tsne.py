import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocess(x, data_name='ihdp'):
    '''
    preprocess features in x
    '''
    if data_name == 'ihdp':
        cons_list = [0, 1, 2, 3, 4, 5]
        disc_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    elif data_name == 'jobs':
        cons_list = [0, 1, 6, 7, 8, 9, 10, 11, 12, 15]
        disc_list = [2, 3, 4, 5, 13, 14, 16]
    elif data_name == 'synthetic':
        cons_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        disc_list = []
    else:
        return x
    cons_x = x[:, cons_list]
    disc_x = x[:, disc_list]
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    trans_cons_x = scaler.fit_transform(cons_x)
    new_x = np.concatenate([trans_cons_x, disc_x], axis=1)
    return new_x


def load_data(name='jobs'):
    if name == 'jobs':
        dat = np.load('../data/jobs/jobs_DW_bin.train.npz')
    elif name == 'ihdp':
        dat = np.load('../data/ihdp/ihdp_npci_1-100.train.npz')
    elif name == 'twins':
        dat = np.load('../data/twins/twins_data.test.npz')
    elif name == 'synthetic':
        dat = np.load('../data/synthetic/syn_train.npz')
    else:
        dat = None

    t = dat['t'][:, 0]
    x = dat['x'][:, :, 0]

    norm_x = preprocess(x, name)

    return norm_x, t, name


def tsne_plt(x, t, name):
    tsne = TSNE(n_components=2, random_state=0)
    x_tsne = tsne.fit_transform(x)
    # x_tsne = MinMaxScaler().fit_transform(x_tsne)

    plt.figure(figsize=(6, 5))
    plt.title('tsne in {}'.format(name))

    plt.scatter(x_tsne[t == 0, 0], x_tsne[t == 0, 1], label='control group', alpha=0.3)
    plt.scatter(x_tsne[t == 1, 0], x_tsne[t == 1, 1], label='treatment group', alpha=0.3)
    plt.legend()
    plt.savefig('~/Desktop/' + name)
    plt.show()


def tsne_plt_3d(x, t, name):
    tsne = TSNE(n_components=3, random_state=0)
    x_tsne = tsne.fit_transform(x)
    x_tsne = MinMaxScaler().fit_transform(x_tsne)

    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(x_tsne[t == 0, 0], x_tsne[t == 0, 1], x_tsne[t == 0, 2], alpha=0.3)
    ax.scatter(x_tsne[t == 1, 0], x_tsne[t == 1, 1], x_tsne[t == 1, 2], alpha=0.3)
    pyplot.show()
    fig.savefig('~/Desktop/' + name)


if __name__ == '__main__':
    x, t, name = load_data('jobs')
    # rep_x = np.load(
    #     '../results/ihdp/results_20210715_212147-937425/reps.npz')
    # rep_x = np.load(
    #     '../results/jobs/results_20210712_100653_030763/reps.npz')
    # rep_x = np.load(
    #     '../results/twins/results_20210717_083738-168258/reps.test.npz')

    rep_x = np.load(
        '../results/twins/results_20210804_224601-915180/reps.test.npz')

    rep = rep_x['rep'][0, 0, :, :]
    tsne_plt(rep, t, name)
    tsne_plt_3d(rep, t, name)
