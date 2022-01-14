import tensorflow as tf
import numpy as np
from cbre.util import *


class CBRENet(object):
    """
    cbre_net implements the cycly-balanced representation learning for counterfactual inference

    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, x, t, y_, p_t, z_norm, flags, r_alpha, r_lambda, r_beta, do_in, do_out, data_x_dim):
        """
        x           The varibales of data
        t           The treatment applied to x, t.shape[1]==1
        y_          The true outcome
        p_t         The treatment probability in all observations
        z_norm      todo unknown
        flags       The arg params
        r_alpha     The coefficient of reconstruction and cycle loss
        r_lambda    The coefficient of regularization of prediction network
        r_beta      The coefficient of gradient penalty in GAN
        do_in       The val of dropout_in
        do_out      The val of dropout_out
        data_x_dim  The dim of varibale x
        """
        self.variables = {}
        # wd_loss: regularization l2 loss
        self.wd_loss = 0

        if flags.nonlin.lower() == 'elu':
            self.nonlin = tf.nn.elu
        else:
            self.nonlin = tf.nn.relu

        self._build_graph(x, t, y_, p_t, z_norm, flags, r_alpha, r_lambda, r_beta, do_in, do_out, data_x_dim)

    def _add_variable(self, var, name):
        """
        Adds variables to the internal track-keeper
        """
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i)  # @TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        """ Create and adds variables to the internal track-keeper """
        # tf.get_variable(name=name, initializer=var)
        var = tf.Variable(var, name=name)
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        """ Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables """
        var = self._create_variable(initializer, name)
        self.wd_loss += wd * tf.nn.l2_loss(var)
        return var

    def _build_graph(self, x, t, y_, p_t, z_norm, flags, r_alpha, r_lambda, r_beta, do_in, do_out, data_x_dim):
        """
        Constructs a TensorFlow subgraph for causal effect inference.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """
        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.r_beta = r_beta
        self.do_in = do_in
        self.do_out = do_out
        self.z_norm = z_norm

        self.encoder_dim = flags.encoder_dim
        encoder_dim = flags.encoder_dim
        self.decoder_dim = flags.decoder_dim
        self.predictor_dim = flags.predictor_dim
        predictor_dim = flags.predictor_dim
        mi_estimator_dim = flags.mi_estimator_dim
        self.discriminator_dim = flags.discriminator_dim
        discriminator_dim = flags.discriminator_dim

        """
        Network Components
        """
        '''
        1. Encoder Network
        '''
        # Construct Encoder network layers, four layers with size 200
        h_rep, h_rep_norm, weights_in = self._build_encoder(x, data_x_dim, flags)

        '''
        2. GAN
        '''
        d0, d1, dp, weights_dis, weights_discore = self._build_adversarial_graph(h_rep_norm, t, encoder_dim,
                                                                                 discriminator_dim, do_out,
                                                                                 flags)
        # discriminator
        # with sigmoid
        # discriminator_loss = tf.reduce_mean(tf.nn.softplus(-d0)) + tf.reduce_mean(tf.nn.softplus(-d1) + d1) + dp
        # without sigmoid
        discriminator_loss = -tf.reduce_mean(d0) + tf.reduce_mean(d1) + r_beta * dp
        # encoder
        # with sigmoid
        # rep_loss = tf.reduce_mean(tf.nn.softplus(-d1))
        # without sigmoid
        # todo rep_loss in paper: rep_loss = tf.reduce_mean(d0) - tf.reduce_mean(d1)
        rep_loss = tf.reduce_mean(d0) - tf.reduce_mean(d1)
        # rep_loss = -tf.reduce_mean(d1)

        '''
        3. Reconstruction 
        '''
        # graph for reconstruction loss
        x0, recons_x_0, x1, recons_x_1 = self._build_reconstruct_graph(x, t, data_x_dim, flags)
        recons_loss = tf.sqrt(tf.reduce_mean(tf.square(x0 - recons_x_0)) + 1.0e-12) + tf.sqrt(
            tf.reduce_mean(tf.square(x1 - recons_x_1)) + 1.0e-12)

        '''
        4. Cycle 
        '''
        x0, cycle_x0, x1, cycle_x1 = self._build_cycle_graph(x, t, data_x_dim, flags)
        cycle_loss = tf.sqrt(tf.reduce_mean(tf.square(x0 - cycle_x0)) + 1.0e-12) + tf.sqrt(
            tf.reduce_mean(tf.square(x1 - cycle_x1)) + 1.0e-12)

        '''
        Predict Networks
        '''
        y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, encoder_dim, predictor_dim, do_out,
                                                                flags)

        """ Compute sample reweighting """
        if flags.reweight_sample:
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * 1 - p_t)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        risk = tf.reduce_mean(sample_weight * tf.square(y_ - y))
        pred_error = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)) + 1.0e-12)

        """ Regularization """
        if flags.p_lambda > 0 and flags.rep_weight_decay:
            for i in range(0, flags.layer_num_encoder):
                if not (flags.varsel and i == 0):  # No penalty on W in variable selection
                    self.wd_loss += tf.nn.l2_loss(weights_in[i])

        """ Total error """
        tot_error = risk

        if flags.p_lambda > 0:
            tot_error = tot_error + r_lambda * self.wd_loss + recons_loss + cycle_loss
        if flags.coef_recons > 0:
            tot_error += flags.coef_recons * recons_loss
        if flags.coef_cycle:
            tot_error += flags.coef_cycle * cycle_loss
        if flags.coef_d:
            tot_error += flags.coef_d * discriminator_loss

        if flags.varsel:
            self.w_proj = tf.placeholder("float", shape=[data_x_dim], name='w_proj')
            self.projection = weights_in[0].assign(self.w_proj)

        self.output = y
        self.tot_loss = tot_error
        self.discriminator_loss = discriminator_loss
        self.rep_loss = rep_loss
        self.rec_loss = recons_loss
        self.cycle_loss = cycle_loss
        self.recons_cycle_loss = recons_loss + cycle_loss
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_dis = weights_dis
        self.weights_discore = weights_discore
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm
        self.dp = dp

    def _build_output_0(self, h_input, encoder_dim, predictor_dim, do_out, flags):
        h_out = [h_input]
        dims = [encoder_dim] + ([predictor_dim] * flags.layer_num_predictor)
        with tf.variable_scope('pred_0') as scope:
            weights_out = []
            biases_out = []

            for i in range(0, flags.layer_num_predictor):
                wo = tf.get_variable(name='w_{}'.format(i),
                                     initializer=tf.random_normal([dims[i], dims[i + 1]],
                                                                  stddev=flags.weight_init / np.sqrt(dims[i])))

                weights_out.append(wo)

                # biases_out.append(tf.Variable(tf.zeros([1, predictor_dim])))
                biases_out.append(tf.get_variable(name='b_{}'.format(i), initializer=tf.zeros([1, predictor_dim])))
                z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

                h_out.append(self.nonlin(z))
                h_out[i + 1] = tf.nn.dropout(h_out[i + 1], do_out)

            weights_pred = self._create_variable(tf.random_normal([predictor_dim, 1],
                                                                  stddev=flags.weight_init / np.sqrt(predictor_dim)),
                                                 'w_pred')
            weights_pred = tf.get_variable(name='w_pred', initializer=tf.random_normal([predictor_dim, 1],
                                                                                       stddev=flags.weight_init / np.sqrt(
                                                                                           predictor_dim)))
            bias_pred = tf.get_variable(initializer=tf.zeros([1]), name='b_pred')

            if flags.varsel or flags.layer_num_predictor == 0:
                self.wd_loss += tf.nn.l2_loss(
                    tf.slice(weights_pred, [0, 0], [predictor_dim - 1, 1]))  # don't penalize treatment coefficient
            else:
                self.wd_loss += tf.nn.l2_loss(weights_pred)

            """ Construct linear classifier """
            h_pred = h_out[-1]
            y = tf.matmul(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred

    def _build_output_1(self, h_input, encoder_dim, predictor_dim, do_out, flags):
        h_out = [h_input]
        dims = [encoder_dim] + ([predictor_dim] * flags.layer_num_predictor)
        with tf.variable_scope('pred_1') as scope:
            weights_out = []
            biases_out = []

            for i in range(0, flags.layer_num_predictor):
                wo = tf.get_variable(name='w_{}'.format(i),
                                     initializer=tf.random_normal([dims[i], dims[i + 1]],
                                                                  stddev=flags.weight_init / np.sqrt(dims[i])))

                weights_out.append(wo)

                # biases_out.append(tf.Variable(tf.zeros([1, predictor_dim])))
                biases_out.append(tf.get_variable(name='b_{}'.format(i), initializer=tf.zeros([1, predictor_dim])))
                z = tf.matmul(h_out[i], weights_out[i]) + biases_out[i]

                h_out.append(self.nonlin(z))
                h_out[i + 1] = tf.nn.dropout(h_out[i + 1], do_out)

            weights_pred = self._create_variable(tf.random_normal([predictor_dim, 1],
                                                                  stddev=flags.weight_init / np.sqrt(predictor_dim)),
                                                 'w_pred')
            weights_pred = tf.get_variable(name='w_pred', initializer=tf.random_normal([predictor_dim, 1],
                                                                                       stddev=flags.weight_init / np.sqrt(
                                                                                           predictor_dim)))
            bias_pred = tf.get_variable(initializer=tf.zeros([1]), name='b_pred')

            if flags.varsel or flags.layer_num_predictor == 0:
                self.wd_loss += tf.nn.l2_loss(
                    tf.slice(weights_pred, [0, 0], [predictor_dim - 1, 1]))  # don't penalize treatment coefficient
            else:
                self.wd_loss += tf.nn.l2_loss(weights_pred)

            """ Construct linear classifier """
            h_pred = h_out[-1]
            y = tf.matmul(h_pred, weights_pred) + bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, encoder_dim, predictor_dim, do_out, flags):
        """ Construct output/regression layers """

        if flags.split_output:

            i0 = tf.to_int32(tf.where(t < 1)[:, 0])
            i1 = tf.to_int32(tf.where(t > 0)[:, 0])

            rep0 = tf.gather(rep, i0)
            rep1 = tf.gather(rep, i1)

            y0, weights_out0, weights_pred0 = self._build_output_0(rep0, encoder_dim, predictor_dim, do_out, flags)
            y1, weights_out1, weights_pred1 = self._build_output_1(rep1, encoder_dim, predictor_dim, do_out, flags)

            y = tf.dynamic_stitch([i0, i1], [y0, y1])
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat(1, [rep, t])
            # y, weights_out, weights_pred = self._build_output(h_input, encoder_dim + 1, predictor_dim, do_out, flags)
            y, weights_out, weights_pred = None, None, None

        return y, weights_out, weights_pred

    def _build_encoder(self, x, data_x_dim, flags):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE) as scope:
            weights_in = []
            biases_in = []

            if flags.batch_norm:
                bn_biases = []
                bn_scales = []

            h_in = [x]

            for i in range(0, flags.layer_num_encoder):
                if i == 0:
                    """ If using variable selection, first layer is just rescaling"""
                    if flags.varsel:
                        weights_in.append(tf.get_variable(name='wg_{}'.format(i),
                                                          initializer=1.0 / data_x_dim * tf.ones([data_x_dim])))
                    else:
                        wg = tf.get_variable(name='wg_{}'.format(i),
                                             initializer=tf.random_normal([data_x_dim, self.encoder_dim],
                                                                          stddev=flags.weight_init / np.sqrt(
                                                                              data_x_dim)))
                        weights_in.append(wg)
                else:
                    wg = tf.get_variable(name='wg_{}'.format(i),
                                         initializer=tf.random_normal([self.encoder_dim, self.encoder_dim],
                                                                      stddev=flags.weight_init / np.sqrt(
                                                                          self.encoder_dim)))
                    weights_in.append(wg)

                biases_in.append(tf.get_variable(name='bi_{}'.format(i), initializer=tf.zeros([1, self.encoder_dim])))
                # z equals outcome of each layer in Encoder Network.
                z = tf.matmul(h_in[i], weights_in[i]) + biases_in[i]

                if flags.batch_norm:
                    batch_mean, batch_var = tf.nn.moments(z, [0])

                    if flags.normalization == 'bn_fixed':
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, 0, 1, 1e-3)
                    else:
                        # bn_biases.append(tf.Variable(tf.zeros([self.encoder_dim])))
                        bn_biases.append(
                            tf.get_variable(name='bn_b_{}'.format(i), initializer=tf.zeros([self.encoder_dim])))
                        # bn_scales.append(tf.Variable(tf.ones([self.encoder_dim])))
                        bn_scales.append(
                            tf.get_variable(name='bn_s_{}'.format(i), initializer=tf.ones([self.encoder_dim])))
                        z = tf.nn.batch_normalization(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                h_in.append(self.nonlin(z))
                h_in[i + 1] = tf.nn.dropout(h_in[i + 1], self.do_in)

            h_rep = h_in[-1]

            # todo normalization meaning?
            if flags.normalization == 'divide':
                h_rep_norm = h_rep / safe_sqrt(tf.reduce_sum(tf.square(h_rep), axis=1, keep_dims=True) + 1.0e-12)
            else:
                h_rep_norm = 1.0 * h_rep
            return h_rep, h_rep_norm, weights_in

    def _build_decoder(self, h_rep, data_x_dim, flags, suffix='0'):
        with tf.variable_scope('decoder_' + suffix, reuse=tf.AUTO_REUSE) as scope:
            weights_in = []
            biases_in = []
            recons_x = [h_rep]
            decoder_dim = flags.decoder_dim
            for i in range(0, flags.layer_num_decoder):
                if i == 0:
                    weights_in.append(tf.get_variable(name='wg_{}'.format(i),
                                                      initializer=tf.random_normal([flags.encoder_dim, decoder_dim],
                                                                                   stddev=flags.weight_init / np.sqrt(
                                                                                       flags.encoder_dim))))
                    biases_in.append(tf.get_variable(name='bi_{}'.format(i), initializer=tf.zeros([1, decoder_dim])))
                elif i == flags.layer_num_decoder - 1:
                    weights_in.append(
                        tf.get_variable(name='wg_{}'.format(i), initializer=tf.random_normal([decoder_dim, data_x_dim],
                                                                                             stddev=flags.weight_init / np.sqrt(
                                                                                                 decoder_dim))))
                    biases_in.append(tf.get_variable(name='bi_{}'.format(i), initializer=tf.zeros([1, data_x_dim])))

                else:
                    weights_in.append(
                        tf.get_variable(name='wg_{}'.format(i), initializer=tf.random_normal([decoder_dim, decoder_dim],
                                                                                             stddev=flags.weight_init / np.sqrt(
                                                                                                 decoder_dim))))
                    biases_in.append(tf.get_variable(name='bi_{}'.format(i), initializer=tf.zeros([1, decoder_dim])))

                # z equals outcome of each layer in Encoder Network.
                z = tf.matmul(recons_x[i], weights_in[i]) + biases_in[i]

                recons_x.append(self.nonlin(z))
                recons_x[i + 1] = tf.nn.dropout(recons_x[i + 1], self.do_in)

            recons_x = recons_x[-1]
            return recons_x, weights_in

    def _build_discriminator_graph_mine(self, x, hrep, data_x_dim, encoder_dim, mi_estimator_dim, flags):
        """ Construct MI estimation layers """
        # two layers with size 200
        with tf.variable_scope('gmi') as scope:
            input_num = tf.shape(x)[0]
            x_shuffle = tf.random_shuffle(x)
            x_conc = tf.concat([x, x_shuffle], axis=0)
            y_conc = tf.concat([hrep, hrep], axis=0)

            # forward
            # [25, 200]
            weights_mi_x = self._create_variable(tf.random_normal([data_x_dim, mi_estimator_dim],
                                                                  stddev=flags.weight_init / np.sqrt(data_x_dim)),
                                                 'weights_mi_x')
            biases_mi_x = self._create_variable(tf.zeros([1, mi_estimator_dim]), 'biases_mi_x')
            # [, 200]
            lin_x = tf.matmul(x_conc, weights_mi_x) + biases_mi_x
            # [200, 200]
            weights_mi_y = self._create_variable(tf.random_normal([encoder_dim, mi_estimator_dim],
                                                                  stddev=flags.weight_init / np.sqrt(encoder_dim)),
                                                 'weights_mi_y')
            biases_mi_y = self._create_variable(tf.zeros([1, mi_estimator_dim]), 'biases_mi_y')
            # [, 200]
            lin_y = tf.matmul(y_conc, weights_mi_y) + biases_mi_y

            # lin_conc = tf.nn.relu(lin_x + lin_y)
            lin_conc = self.nonlin(lin_x + lin_y)

            weights_mi_pred = self._create_variable(tf.random_normal([mi_estimator_dim, 1],
                                                                     stddev=flags.weight_init / np.sqrt(
                                                                         mi_estimator_dim)),
                                                    'gmi_p')
            biases_mi_pred = self._create_variable(tf.zeros([1, mi_estimator_dim]), 'biases_mi_pred')
            gmi_output = tf.matmul(lin_conc, weights_mi_pred) + biases_mi_pred
            # real estimator outcome: shape=[input_num, 1]
            real_estimate = gmi_output[:input_num]
            # fake estimator outcome: shape=[input_num, 1]
            fake_estimate = gmi_output[input_num:]

        return real_estimate, fake_estimate, weights_mi_x, weights_mi_y, weights_mi_pred

    def _build_discriminator_adversarial(self, hrep, encoder_dim, discriminator_dim, do_out, flags):
        """ Construct adversarial discriminator layers """
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE) as scope:
            h_dis = [hrep]

            weights_dis = []
            biases_dis = []
            for i in range(0, flags.layer_num_discriminator):

                if i == 0:
                    weights_dis.append(tf.get_variable(name='wg_{}'.format(i),
                                                       initializer=tf.random_normal([encoder_dim, discriminator_dim],
                                                                                    stddev=flags.weight_init / np.sqrt(
                                                                                        encoder_dim))))
                else:
                    weights_dis.append(tf.get_variable(name='wg_{}'.format(i), initializer=tf.random_normal(
                        [discriminator_dim, discriminator_dim],
                        stddev=flags.weight_init / np.sqrt(
                            discriminator_dim))))
                biases_dis.append(tf.get_variable(name='bi_{}'.format(i), initializer=tf.zeros([1, discriminator_dim])))
                z = tf.matmul(h_dis[i], weights_dis[i]) + biases_dis[i]
                h_dis.append(self.nonlin(z))
                h_dis[i + 1] = tf.nn.dropout(h_dis[i + 1], do_out)

            weights_discore = tf.get_variable(initializer=tf.random_normal([discriminator_dim, 1],
                                                                           stddev=flags.weight_init / np.sqrt(
                                                                               discriminator_dim)), name='dc_p')
            bias_dc = tf.get_variable(initializer=tf.zeros([1]), name='dc_b_p')

            h_score = h_dis[-1]
            dis_score = tf.matmul(h_score, weights_discore) + bias_dc

        return dis_score, weights_dis, weights_discore

    def _build_adversarial_graph(self, rep, t, encoder_dim, discriminator_dim, do_out, flags):
        """
        Construct adversarial discriminator
        """
        # three layers with size 200

        i0 = tf.to_int32(tf.where(t < 1)[:, 0])
        i1 = tf.to_int32(tf.where(t > 0)[:, 0])

        rep0 = tf.gather(rep, i0)
        rep1 = tf.gather(rep, i1)

        z_rep0 = tf.reduce_max(rep0, axis=0, keep_dims=True)
        z_rep1 = tf.reduce_max(rep1, axis=0, keep_dims=True)

        z_rep0_conc = tf.concat([z_rep0, self.z_norm], axis=1)
        z_rep1_conc = tf.concat([z_rep1, self.z_norm], axis=1)

        d0, weights_dis, weights_discore = self._build_discriminator_adversarial(z_rep0_conc, encoder_dim + encoder_dim,
                                                                                 discriminator_dim,
                                                                                 do_out, flags)
        d1, weights_dis, weights_discore = self._build_discriminator_adversarial(z_rep1_conc, encoder_dim + encoder_dim,
                                                                                 discriminator_dim,
                                                                                 do_out, flags)

        # gradient penalty
        alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
        alpha = alpha_dist.sample((1, 1))
        interpolated = z_rep1 + alpha * (z_rep0 - z_rep1)
        interpolated_conc = tf.concat([interpolated, self.z_norm], axis=1)
        inte_logit, weights_dis, weights_discore = self._build_discriminator_adversarial(interpolated_conc,
                                                                                         encoder_dim + encoder_dim,
                                                                                         discriminator_dim, do_out,
                                                                                         flags)
        gradients = tf.gradients(inte_logit, [interpolated])[0]
        grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]) + 1.0e-12)
        gradient_penalty = tf.reduce_mean(tf.square(grad_l2 - 1.0))

        return d0, d1, gradient_penalty, weights_dis, weights_discore

    def _build_reconstruct_graph(self, x, t, data_x_dim, flags):
        """ construct graph for later computing reconstruction loss easily

        Parameters:
        x   The varibales of data
        t   The treatment applied to x

        Returns:
        x0  x[t=0]
        reconstruct_x   reconstruct x when pass encoder and decoder networks
        """
        i0 = tf.to_int32(tf.where(t < 1)[:, 0])
        i1 = tf.to_int32(tf.where(t > 0)[:, 0])

        x0 = tf.gather(x, i0)
        x1 = tf.gather(x, i1)
        h_rep_0, h_rep_norm_0, weights_in_0 = self._build_encoder(x0, data_x_dim, flags)
        h_rep_1, h_rep_norm_1, weights_in_1 = self._build_encoder(x1, data_x_dim, flags)

        recons_x_0, _ = self._build_decoder(h_rep_norm_0, data_x_dim, flags, suffix='0')
        recons_x_1, _ = self._build_decoder(h_rep_norm_1, data_x_dim, flags, suffix='1')
        return x0, recons_x_0, x1, recons_x_1

    def _build_cycle_graph(self, x, t, data_x_dim, flags):
        """ construct graph for later computing cycle loss easily

        Parameters:
        x   The varibales of data
        t   The treatment applied to x

        Returns:
        x0  x[t=0]
        reconstruct_x   reconstruct x when pass encoder and decoder networks
        """
        i0 = tf.to_int32(tf.where(t < 1)[:, 0])
        i1 = tf.to_int32(tf.where(t > 0)[:, 0])

        x0 = tf.gather(x, i0)
        x1 = tf.gather(x, i1)
        # cycle x0-x1'-x0
        _, h_rep_norm_0, _ = self._build_encoder(x0, data_x_dim, flags)
        temp_x_0_in_1, _ = self._build_decoder(h_rep_norm_0, data_x_dim, flags, suffix='1')
        _, cyc_h_rep_norm_0, _ = self._build_encoder(temp_x_0_in_1, data_x_dim, flags)
        cycle_x0, _ = self._build_decoder(cyc_h_rep_norm_0, data_x_dim, flags, suffix='0')

        # cycle x1-x0'-x1
        _, h_rep_norm_1, _ = self._build_encoder(x1, data_x_dim, flags)
        temp_x_1_in_0, _ = self._build_decoder(h_rep_norm_1, data_x_dim, flags, suffix='0')
        _, cyc_h_rep_norm_1, _ = self._build_encoder(temp_x_1_in_0, data_x_dim, flags)
        cycle_x1, _ = self._build_decoder(cyc_h_rep_norm_1, data_x_dim, flags, suffix='1')

        return x0, cycle_x0, x1, cycle_x1
