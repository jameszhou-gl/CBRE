import tensorflow as tf
import numpy as np
import os
import sys
import math
import getopt
import random
import datetime
import traceback
from cbre.cbre_net import CBRENet
from cbre.util import *
from sklearn import metrics

# Define parameter flags
flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('layer_num_encoder', 2, """layer numbers of encoder network. """)
tf.app.flags.DEFINE_integer('layer_num_decoder', 2, """layer numbers of decoder network. """)
tf.app.flags.DEFINE_integer('layer_num_predictor', 2, """layer numbers of predictor network. """)
tf.app.flags.DEFINE_integer('layer_num_discriminator', 2, """layer numbers of discriminator network. """)
tf.app.flags.DEFINE_float('coef_recons', 1.0, """coefficient of reconstruct loss""")
tf.app.flags.DEFINE_float('coef_cycle', 1.0, """coefficient of cycle loss""")
tf.app.flags.DEFINE_float('coef_d', 1.0, """coefficient of discriminator loss""")
tf.app.flags.DEFINE_float('p_alpha', 0.0, """Weight of recons+cycle loss """)
tf.app.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_float('p_beta', 10.0, """Gradient penalty weight. """)
tf.app.flags.DEFINE_float('thres_d', 5, """threshold for training discriminator network""")
tf.app.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.app.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('encoder_dim', 100, """Layer dimensions of Encoder network""")
tf.app.flags.DEFINE_integer('decoder_dim', 100, """Layer dimensions of Decoder network""")
tf.app.flags.DEFINE_integer('predictor_dim', 100, """Predictor layer dimensions. """)
tf.app.flags.DEFINE_integer('mi_estimator_dim', 100, """MI estimation layer dimensions. """)
tf.app.flags.DEFINE_integer('discriminator_dim', 100, """Discriminator layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'none',
                           """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.app.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_string('outdir', '../results/ihdp/', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', '../data/topic/csv/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_integer('output_csv', 0, """Whether to save a CSV file with the results""")
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.app.flags.DEFINE_integer('pred_output_delay', -1,
                            """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.app.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.app.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1,
                            """Whether to reweight sample for prediction loss with average treatment probability. """)

if flags.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100
EARLY_STOP_DELTA = 0.05
IF_EARLY_STOP = False


def early_stop(valid_obj, min_valid_loss, patience_cnt):
    if valid_obj - min_valid_loss > EARLY_STOP_DELTA:
        patience_cnt += 1
    else:
        min_valid_loss = valid_obj
    return patience_cnt, min_valid_loss


def train(rbnet, sess, train_step, train_discriminator_step, train_rec_step, train_encoder_step,
          train_pred_step, data_exp,
          valid_index,
          test_data_exp,
          logfile, i_exp):
    """ Trains a rbnet model on supplied data """

    ''' Train/validation split '''
    data_num = data_exp['x'].shape[0]
    range_of_data_num = range(data_num)
    train_index = list(set(range_of_data_num) - set(valid_index))
    train_num = len(train_index)
    valid_num = len(valid_index)

    ''' Compute treatment probability'''
    p_treated = np.mean(data_exp['t'][train_index, :])

    z_norm = np.random.normal(0., 1., (1, flags.encoder_dim))

    ''' Set up loss feed_dicts'''
    # dict_factual means in train_data
    dict_factual = {rbnet.x: data_exp['x'][train_index, :], rbnet.t: data_exp['t'][train_index, :],
                    rbnet.y_: data_exp['yf'][train_index, :],
                    rbnet.do_in: flags.dropout_in, rbnet.do_out: flags.dropout_out, rbnet.r_lambda: flags.p_lambda,
                    rbnet.r_beta: flags.p_beta,
                    rbnet.p_t: p_treated, rbnet.z_norm: z_norm}

    if flags.val_part > 0:
        dict_valid = {rbnet.x: data_exp['x'][valid_index, :],
                      rbnet.t: data_exp['t'][valid_index, :],
                      rbnet.y_: data_exp['yf'][valid_index, :],
                      rbnet.do_in: flags.dropout_in, rbnet.do_out: flags.dropout_out, rbnet.r_lambda: flags.p_lambda,
                      rbnet.r_beta: flags.p_beta,
                      rbnet.p_t: p_treated, rbnet.z_norm: z_norm}
    else:
        dict_valid = dict()

    if data_exp['HAVE_TRUTH']:
        dict_cfactual = {rbnet.x: data_exp['x'][train_index, :], rbnet.t: 1 - data_exp['t'][train_index, :],
                         rbnet.y_: data_exp['ycf'][train_index, :],
                         rbnet.do_in: flags.dropout_in, rbnet.do_out: flags.dropout_out, rbnet.z_norm: z_norm}
    else:
        dict_cfactual = dict()

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    objnan = False
    losses = []
    reps = []
    reps_test = []
    log(logfile, 'train num: {}, valid num: {}'.format(train_num, valid_num))
    # for early-stop in valid_loss
    PATIENCE_THRES = 5
    min_valid_loss = 50000
    patience_cnt = 0
    ''' Train for multiple iterations '''
    for i in range(flags.iterations):
        ''' Fetch sample '''
        I = list(range(0, train_num))
        np.random.shuffle(I)

        for i_batch in range(train_num // flags.batch_size):
            if i_batch < train_num // flags.batch_size - 1:
                I_b = I[i_batch * flags.batch_size:(i_batch + 1) * flags.batch_size]
            else:
                I_b = I[i_batch * flags.batch_size:]
            x_batch = data_exp['x'][train_index, :][I_b, :]
            t_batch = data_exp['t'][train_index, :][I_b]
            y_batch = data_exp['yf'][train_index, :][I_b]

            z_norm_batch = np.random.normal(0., 1., (1, flags.encoder_dim))

            ''' Do one step of gradient descent '''
            if not objnan:
                # for sub_dc in range(0, 3):
                ''' train discriminator
                '''
                discriminator_loss = sess.run(rbnet.discriminator_loss, feed_dict=dict_factual)
                if np.abs(discriminator_loss) < flags.thres_d:
                    sess.run(train_discriminator_step,
                             feed_dict={rbnet.x: x_batch, rbnet.t: t_batch, rbnet.r_beta: flags.p_beta,
                                        rbnet.do_in: flags.dropout_in, rbnet.do_out: flags.dropout_out,
                                        rbnet.z_norm: z_norm_batch})

                # train tot
                sess.run(train_step, feed_dict={rbnet.x: x_batch, rbnet.t: t_batch,
                                                rbnet.y_: y_batch, rbnet.do_in: flags.dropout_in,
                                                rbnet.do_out: flags.dropout_out,
                                                rbnet.r_lambda: flags.p_lambda, rbnet.p_t: p_treated,
                                                rbnet.r_beta: flags.p_beta, rbnet.z_norm: z_norm_batch})

        ''' Compute predictions every M iterations '''
        if (flags.pred_output_delay > 0 and i % flags.pred_output_delay == 0) or i == flags.iterations - 1:

            y_pred_f = sess.run(rbnet.output, feed_dict={rbnet.x: data_exp['x'],
                                                         rbnet.t: data_exp['t'], rbnet.do_in: flags.dropout_in,
                                                         rbnet.do_out: flags.dropout_out})
            y_pred_cf = sess.run(rbnet.output, feed_dict={rbnet.x: data_exp['x'],
                                                          rbnet.t: 1 - data_exp['t'], rbnet.do_in: flags.dropout_in,
                                                          rbnet.do_out: flags.dropout_out})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf), axis=1))
            # print(np.array(preds_train).shape)
            if test_data_exp:
                y_pred_f_test = sess.run(rbnet.output, feed_dict={rbnet.x: test_data_exp['x'],
                                                                  rbnet.t: test_data_exp['t'],
                                                                  rbnet.do_in: flags.dropout_in,
                                                                  rbnet.do_out: flags.dropout_out})
                y_pred_cf_test = sess.run(rbnet.output, feed_dict={rbnet.x: test_data_exp['x'],
                                                                   rbnet.t: 1 - test_data_exp['t'],
                                                                   rbnet.do_in: flags.dropout_in,
                                                                   rbnet.do_out: flags.dropout_out})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test), axis=1))

            if flags.save_rep and i_exp == 1:
                reps_i = sess.run([rbnet.h_rep], feed_dict={rbnet.x: data_exp['x'],
                                                            rbnet.do_in: flags.dropout_in,
                                                            rbnet.do_out: flags.dropout_out})
                reps.append(reps_i)

                if test_data_exp:
                    reps_test_i = sess.run([rbnet.h_rep], feed_dict={rbnet.x: test_data_exp['x'],
                                                                     rbnet.do_in: flags.dropout_in,
                                                                     rbnet.do_out: flags.dropout_out})
                    reps_test.append(reps_test_i)
        ''' Compute loss every N iterations '''
        if i % flags.output_delay == 0 or i == flags.iterations - 1:
            obj_loss, f_error, recons_err, cycle_err, discriminator_loss, rep_loss, gradient_pen = \
                sess.run(
                    [rbnet.tot_loss, rbnet.pred_loss, rbnet.rec_loss, rbnet.cycle_loss, rbnet.discriminator_loss,
                     rbnet.rep_loss,
                     rbnet.dp],
                    feed_dict=dict_factual)

            cf_error = np.nan
            if data_exp['HAVE_TRUTH']:
                cf_error = sess.run(rbnet.pred_loss, feed_dict=dict_cfactual)

            if flags.val_part > 0:
                valid_obj, valid_f_error, valid_recons, valid_cycle, valid_dc, valid_rep_r, valid_dp = \
                    sess.run(
                        [rbnet.tot_loss, rbnet.pred_loss, rbnet.rec_loss, rbnet.cycle_loss,
                         rbnet.discriminator_loss,
                         rbnet.rep_loss,
                         rbnet.dp],
                        feed_dict=dict_valid)
            else:
                valid_obj = valid_f_error = valid_recons = valid_cycle = valid_dc = valid_rep_r = valid_dp = np.nan

            losses.append(
                [obj_loss, f_error, cf_error, recons_err, cycle_err, discriminator_loss, rep_loss, gradient_pen,
                 valid_f_error, valid_recons, valid_cycle, valid_dc, valid_rep_r, valid_dp, valid_obj])
            train_loss_st = 'iter: {}. Train: tot_loss: {:.3f}, pred_loss: {:.3f}'.format(
                i, obj_loss, f_error)
            train_loss_st += ', recons: {:.3f}, cycle: {:.3f}, dc_loss: {:.3f}, rep_loss: {:.3f}, cf_error: {:.3f}'.format(
                recons_err, cycle_err,
                discriminator_loss,
                rep_loss, cf_error)
            log(logfile, train_loss_st)

            valid_loss_st = 'iter: {}. Valid: tot_loss: {:.3f}, pred_loss: {:.3f}'.format(
                i, valid_obj, valid_f_error)
            valid_loss_st += ', recons: {:.3f}, cycle: {:.3f}, dc_loss: {:.3f}, rep_loss: {:.3f}'.format(valid_recons,
                                                                                                         valid_cycle,
                                                                                                         valid_dc,
                                                                                                         valid_rep_r)
            log(logfile, valid_loss_st)
            if np.isnan(obj_loss):
                log(logfile, 'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True
            # early-stop in valid_loss
            if IF_EARLY_STOP:
                patience_cnt, min_valid_loss = early_stop(valid_obj, min_valid_loss, patience_cnt)
                # print(patience_cnt)
                # print(min_valid_loss)
                if patience_cnt > PATIENCE_THRES:
                    log(logfile, 'early stoping at iter {}'.format(i))
                    break

    return losses, preds_train, preds_test, reps, reps_test


def run(outdir):
    """ Runs an experiment and stores result in outdir """
    ''' Set up paths and start log '''
    npzfile = os.path.join(outdir, 'result')
    npzfile_test = os.path.join(outdir, 'result.test')
    repfile = os.path.join(outdir, 'reps')
    repfile_test = os.path.join(outdir, 'reps.test')
    outform = os.path.join(outdir, 'y_pred')
    outform_test = os.path.join(outdir, 'y_pred.test')
    lossform = os.path.join(outdir, 'loss')
    logfile = os.path.join(outdir, 'log.txt')
    f = open(logfile, 'w')
    f.close()
    dataform = os.path.join(flags.datadir, flags.dataform)
    dataform_test = os.path.join(flags.datadir, flags.data_test)

    ''' Set random seeds '''
    random.seed(flags.seed)
    tf.set_random_seed(flags.seed)
    np.random.seed(flags.seed)

    ''' Save parameters '''
    save_config(os.path.join(outdir, 'config.txt'))
    log(logfile, 'Training with hyperparameters: beta={:.2g}, lambda={:.2g}'.format(flags.p_beta, flags.p_lambda))

    ''' Load Data '''
    datapath = dataform
    datapath_test = dataform_test
    log(logfile, 'Train data:{}'.format(datapath))
    log(logfile, 'Test data:{}'.format(datapath_test))

    data = load_data(datapath)
    test_data = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [{},{}]'.format(data['n'], data['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x = tf.placeholder("float", shape=[None, data['dim']], name='x')  # Features
    t = tf.placeholder("float", shape=[None, 1], name='t')  # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    # todo what's role of znorm
    znorm = tf.placeholder("float", shape=[None, flags.encoder_dim], name='z_norm')

    ''' Parameter placeholders '''
    # r_alpha is coefficient of reconstruction and cycle loss
    r_alpha = tf.placeholder('float', name='r_alpha')
    # r_lambda is coefficient of regularization of prediction network.
    r_lambda = tf.placeholder("float", name='r_lambda')
    # r_beta is coefficient of gradient penalty in GAN
    r_beta = tf.placeholder("float", name='r_beta')

    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    # treatment probability in all observations
    p = tf.placeholder("float", name='p_treated')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    # dims = [data['dim'], flags.encoder_dim, flags.predictor_dim, flags.mi_estimator_dim, flags.discriminator_dim]
    data_x_dim = data['dim']

    rbnet = CBRENet(x, t, y_, p, znorm, flags, r_alpha, r_lambda, r_beta, do_in, do_out, data_x_dim)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(flags.lrate, global_step,
                                    NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)

    counter_enc = tf.Variable(0, trainable=False)
    lr_enc = tf.train.exponential_decay(flags.lrate, counter_enc,
                                        NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)

    counter_dc = tf.Variable(0, trainable=False)
    lr_dc = tf.train.exponential_decay(flags.lrate, counter_dc,
                                       NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)

    counter_dec = tf.Variable(0, trainable=False)
    lr_dec = tf.train.exponential_decay(flags.lrate, counter_dec,
                                        NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)
    counter_rec = tf.Variable(0, trainable=False)
    lr_rec = tf.train.exponential_decay(flags.lrate, counter_rec,
                                        NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)
    counter_pred = tf.Variable(0, trainable=False)
    lr_pred = tf.train.exponential_decay(flags.lrate, counter_pred,
                                         NUM_ITERATIONS_PER_DECAY, flags.lrate_decay, staircase=True)

    if flags.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
        opt_enc = tf.train.AdamOptimizer(
            learning_rate=lr_enc,
            beta1=0.5,
            beta2=0.9)
        opt_dc = tf.train.AdamOptimizer(
            learning_rate=lr_dc,
            beta1=0.5,
            beta2=0.9)
        opt_dec = tf.train.AdamOptimizer(
            learning_rate=lr_dec,
            beta1=0.5,
            beta2=0.9
        )
        opt_rec = tf.train.AdamOptimizer(
            learning_rate=lr_rec,
            beta1=0.5,
            beta2=0.9
        )
        opt_pred = tf.train.AdamOptimizer(
            learning_rate=lr_pred,
            beta1=0.5,
            beta2=0.9
        )
        # opt_gmi = tf.train.AdamOptimizer(lr_gmi)
    else:
        lr_gan = 5e-5
        opt = tf.train.RMSPropOptimizer(lr_gan)
        opt_enc = tf.train.RMSPropOptimizer(lr_gan)
        opt_dc = tf.train.RMSPropOptimizer(lr_gan)
        opt_dec = tf.train.RMSPropOptimizer(lr_gan)
        opt_rec = tf.train.RMSPropOptimizer(lr_gan)
        opt_pred = tf.train.RMSPropOptimizer(lr_gan)

    ''' Unused gradient clipping '''
    # gvs = opt.compute_gradients(rbnet.tot_loss)
    # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    # train_step = opt.apply_gradients(capped_gvs, global_step=global_step)
    '''
    # var_scope_get
    var_enc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    log(logfile, 'var_enc list: {}'.format([v.name for v in var_enc]))

    var_de = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    log(logfile, 'var_de list: {}'.format([v.name for v in var_de]))
    var_de.extend(var_enc)

    var_dc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    log(logfile, 'var_dc list: {}'.format([v.name for v in var_dc]))
    # var_dc.extend(var_enc)

    var_pred = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
    log(logfile, 'var_pred list: {}'.format([v.name for v in var_pred]))
    var_pred.extend(var_enc)

    train_dec_step = opt_dec.minimize(rbnet.recons_cycle_loss, global_step=counter_dc, var_list=var_de)
    train_discriminator_step = opt_dc.minimize(rbnet.discriminator_loss, global_step=counter_dc, var_list=var_dc)
    train_encoder_step = opt_enc.minimize(rbnet.rep_loss, global_step=counter_enc, var_list=var_enc)
    # todo why train_step using var_pred(pred and enc)?
    train_step = opt.minimize(rbnet.tot_loss, global_step=global_step, var_list=var_pred)
    '''
    # var_scope_get
    var_enc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
    log(logfile, 'var_enc list: {}'.format([v.name for v in var_enc]))

    var_de = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
    log(logfile, 'var_de list: {}'.format([v.name for v in var_de]))
    # var_de.extend(var_enc)

    var_dc = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    log(logfile, 'var_dc list: {}'.format([v.name for v in var_dc]))
    # var_dc.extend(var_enc)

    var_pred = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='pred')
    log(logfile, 'var_pred list: {}'.format([v.name for v in var_pred]))
    # var_pred.extend(var_enc)

    train_discriminator_step = opt_dc.minimize(rbnet.discriminator_loss, global_step=counter_dc, var_list=var_dc)
    train_rec_step = opt_rec.minimize(rbnet.rec_loss, global_step=counter_rec, var_list=var_de)
    train_encoder_step = opt_enc.minimize(rbnet.pred_loss + rbnet.discriminator_loss + rbnet.cycle_loss,
                                          global_step=counter_enc, var_list=var_enc)
    # train_dec_step = opt_dec.minimize(rbnet.recons_cycle_loss, global_step=counter_dc, var_list=var_de)
    train_pred_step = opt_pred.minimize(rbnet.pred_loss, global_step=counter_pred, var_list=var_pred)
    var_pred.extend(var_enc)
    train_step = opt.minimize(rbnet.tot_loss, global_step=global_step, var_list=var_pred)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []

    ''' Handle repetitions '''
    n_experiments = flags.experiments
    if flags.repetitions > 1:
        if flags.experiments > 1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = flags.repetitions

    ''' Run for all repeated experiments '''
    data_exp = dict()
    test_data_exp = dict()
    for i_exp in range(1, n_experiments + 1):
        log(logfile, 'Training on experiment {}/{}...'.format(i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''
        if i_exp == 1 or flags.experiments > 1:
            data_exp['x'] = data['x'][:, :, i_exp - 1]
            data_exp['x'] = preprocess(data_exp['x'], data_name=flags.dataform.split('_')[0])
            data_exp['t'] = data['t'][:, i_exp - 1:i_exp]
            data_exp['yf'] = data['yf'][:, i_exp - 1:i_exp]
            if data['HAVE_TRUTH']:
                data_exp['ycf'] = data['ycf'][:, i_exp - 1:i_exp]
            else:
                data_exp['ycf'] = None

            test_data_exp['x'] = test_data['x'][:, :, i_exp - 1]
            test_data_exp['t'] = test_data['t'][:, i_exp - 1:i_exp]
            test_data_exp['yf'] = test_data['yf'][:, i_exp - 1:i_exp]
            if test_data['HAVE_TRUTH']:
                test_data_exp['ycf'] = test_data['ycf'][:, i_exp - 1:i_exp]
            else:
                test_data_exp['ycf'] = None

            data_exp['HAVE_TRUTH'] = data['HAVE_TRUTH']
            test_data_exp['HAVE_TRUTH'] = test_data['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        _, valid_index = validation_split(data_exp, flags.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
            train(rbnet, sess, train_step, train_discriminator_step, train_rec_step, train_encoder_step,
                  train_pred_step, data_exp,
                  valid_index,
                  test_data_exp, logfile, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_exp, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train, 1, 3), 0, 2)
        # out_preds_train = all_preds_train
        out_preds_test = np.swapaxes(np.swapaxes(all_preds_test, 1, 3), 0, 2)
        # out_preds_test = all_preds_test
        # print(all_losses)
        out_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)
        # out_losses = all_losses

        ''' Store predictions '''
        log(logfile, 'Saving result to {}...\n'.format(outdir))

        ''' Save results and predictions '''
        all_valid.append(valid_index)

        np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if flags.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            np.savez(repfile_test, rep=reps_test)
    log(logfile, '\ntrain run done\n')


def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = os.path.join(flags.outdir, 'results_' + timestamp)
    os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir + 'error.txt', 'w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise


if __name__ == '__main__':
    tf.app.run()
