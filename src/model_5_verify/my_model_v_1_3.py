import operator
import pickle
import math
import tensorflow as tf
import numpy as np
import glob
import pandas as pd
import os

from numpy.core._multiarray_umath import ndarray
from sklearn.manifold import TSNE
import sys
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import time
import inspect
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import yaml

class model:
    def __init__(self, MODEL_NAME, SAVE_DIR, OP_DIR):

        self.inference = False
        self.save_dir = SAVE_DIR
        self.op_dir = OP_DIR
        self.frozen_file = None
        self.ts = None
        self.use_bias = False
        self.save_loss_fig = True
        self.show_figure = False
        self.use_activation = False
        self.test_SerialID = None
        self._model_name = MODEL_NAME
        return

    def set_model_hyperparams(
            self,
            domain_dims,
            emb_dims,
            use_bias=False,
            batch_size=128,
            num_epochs=20,
            learning_rate=0.001,
            alpha=0.0025
    ):
        MODEL_NAME = self._model_name
        self.learning_rate = learning_rate
        self.num_domains = len(domain_dims)
        self.domain_dims = domain_dims
        self.num_emb_layers = len(emb_dims)
        self.emb_dims = emb_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_signature = MODEL_NAME + '_'.join([str(e) for e in emb_dims])
        self.use_bias = use_bias
        self.alpha = alpha
        return

    def set_model_options(
            self,
            show_loss_figure,
            save_loss_figure,
            set_w_mean=True
    ):
        self.show_loss_figure = show_loss_figure
        self.save_loss_figure = save_loss_figure
        self.set_w_mean = True
        return
    def set_SerialID(self, test_SerialID):
        self.test_SerialID = test_SerialID

    def get_weight_variable(
            self,
            shape,
            name=None
    ):
        initializer = tf.contrib.layers.xavier_initializer()
        if name is not None:
            return tf.Variable(initializer(shape), name=name)
        else:
            return tf.Variable(initializer(shape))

    def define_wbs(self):
        # print('>> Defining weights :: start')

        self.W = [None] * self.num_emb_layers
        self.b = [None] * self.num_emb_layers

        wb_scope_name = 'params'
        # doman dimensions

        layer_1_dims = []
        for i in self.domain_dims:
            _d = int(math.ceil(math.log(i, 2)))
            if _d <= 1:
                _d += 1
            layer_1_dims.append(_d)
        # print(layer_1_dims)

        with tf.name_scope(wb_scope_name):
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            self.wb_names = []

            # -------
            # For each layer define W , b
            # -------
            for l in range(self.num_emb_layers):
                self.W[l] = [None] * self.num_domains
                self.b[l] = [None] * self.num_domains

                # print("----> Layer", (l + 1))
                if l == 0:
                    layer_inp_dims = self.domain_dims
                    # layer_op_dims = [self.emb_dims[l]] * self.num_domains
                    layer_op_dims = layer_1_dims
                    # print(layer_inp_dims)
                    # print(layer_op_dims)
                else:
                    if l == 1:
                        layer_inp_dims = layer_1_dims
                    else:
                        layer_inp_dims = [self.emb_dims[l - 1]] * self.num_domains
                    layer_op_dims = [self.emb_dims[l]] * self.num_domains
                    # print(layer_inp_dims)
                    # print(layer_op_dims)
                for d in range(self.num_domains):
                    # print('-> Domain', (d + 1))
                    _name = 'W_' + str(l) + str(d)

                    if self.inference is True:
                        n = prefix + _name + ':0'
                        self.W[l][d] = self.restore_graph.get_tensor_by_name(n)
                    else:

                        z = self.get_weight_variable(
                            [layer_inp_dims[d],
                             layer_op_dims[d]],
                            name=_name)
                        self.W[l][d] = z
                        self.wb_names.append(prefix + _name)
                if self.use_bias:
                    for d in range(self.num_domains):
                        _name_b = 'bias_b_' + str(l) + str(d)
                        b_dims = [layer_op_dims[d]]  # opdim1, opdim 2

                        if self.inference is True:
                            n = prefix + _name_b + ':0'
                            self.b[l][d] = self.restore_graph.get_tensor_by_name(n)
                        else:
                            z = self.get_weight_variable(b_dims, _name_b)
                            self.b[l][d] = z
                            self.wb_names.append(prefix + _name_b)
        # print(self.wb_names)
        # print('>> Defining weights :: end')

    def restore_model(self):

        # Model already restored!
        if self.inference is True:
            return

        self.inference = True
        if self.frozen_file is None:
            # ensure embedding dimensions are correct
            emb = '_'.join([str(_) for _ in self.emb_dims])
            files = glob.glob(os.path.join(self.save_dir, 'checkpoints', '*' + emb + '*.pb'))
            f_name = files[-1]
            self.frozen_file = f_name

        if self.ts is None:
            self.ts = '.'.join(
                (''.join(
                    self.frozen_file.split('_')[-1]
                )
                ).split('.')[:1])
        # print('ts ::', self.ts)

        tf.reset_default_graph()
        # print('Frozen file', self.frozen_file)

        with tf.gfile.GFile(self.frozen_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.restore_graph = None

        with tf.Graph().as_default() as g:
            try:
                tf.graph_util.import_graph_def(
                    graph_def,
                    input_map=None,
                    name='',
                    return_elements=None,
                    op_dict=None,
                    producer_op_list=None
                )
            except:
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    name='',
                    return_elements=None,
                    op_dict=None,
                    producer_op_list=None
                )
            self.restore_graph = g
            self.inference = True
            self.build_model()
        return

    # ---------------------------------------------------------- #
    def build_model(self):
        # print('Building model : start ')
        self.model_scope_name = 'model'

        with tf.variable_scope(self.model_scope_name):
            # batch_size ,domains, label_id
            self.x_pos_inp = tf.placeholder(
                tf.int32,
                [None, self.num_domains],
                name='x_pos_samples'
            )

            # Inside the scope	- define the weights and biases
            self.define_wbs()

            x_pos_inp = tf.split(
                self.x_pos_inp,
                self.num_domains,
                axis=-1
            )
            x_pos_WXb = [None] * self.num_domains

            for d in range(self.num_domains):
                # for each domain
                prev = None
                for l in range(self.num_emb_layers):
                    # print("----> Layer", (l + 1))
                    if l == 0:
                        a = tf.nn.embedding_lookup(
                            self.W[l][d],
                            x_pos_inp[d]
                        )
                        _wx = tf.squeeze(a, axis=1)
                    else:
                        _x = prev
                        _wx = tf.matmul(
                            _x,
                            self.W[l][d]
                        )

                    if self.use_bias:
                        _wx_b = tf.add(_wx, self.b[l][d])
                    else:
                        _wx_b = _wx

                    prev = _wx_b
                x_pos_WXb[d] = prev
                # print(x_pos_WXb[d].shape)

            emb_op_pos = x_pos_WXb
            self.joint_emb_op = tf.stack(emb_op_pos, axis=1)
            self.concat_emb_op = tf.concat(emb_op_pos, axis=1)

            # Calculate the combined embedding of the record

            self.mean_emb_op = tf.reduce_mean(
                self.joint_emb_op,
                axis=1,
                keepdims=False
            )

            if self.inference:
                if self.set_w_mean:
                    self.entity_prob_x = tf.placeholder(
                        tf.float32,
                        [None, self.num_domains],
                        name='entity_prob_x'
                    )

                    # self.domain_alpha = tf.placeholder(
                    #     tf.float32,
                    #     [self.num_domains],
                    #     name='domain_alpha'
                    # )

                    _ep_x = tf.divide(self.entity_prob_x, self.alpha)
                    _ep_x = tf.add(_ep_x, 1)
                    _ep_x = tf.pow(_ep_x, -1)

                    # _ep_x = tf.add(-self.entity_prob_x, 1)

                    self.ep_x = _ep_x

                    # emb_op_pos : shape [ batch, domains, emd_dim]
                    # entity_prob_x : shape [batch, domains]
                    ep_x = tf.reshape(
                        self.ep_x,
                        [-1, self.ep_x.shape[-1], 1]
                    )
                    # print('ep_x shape', ep_x.shape)
                    ep_x = tf.tile(
                        ep_x,
                        [1, 1, self.emb_dims[-1]]
                    )
                    # print('ep_x shape', ep_x.shape)

                    # --------------------------- #
                    # apply  weight : element wise multiply
                    w_joint_emb = tf.multiply(self.joint_emb_op, ep_x)

                    self.w_mean_emb_op = tf.reduce_mean(
                        w_joint_emb,
                        axis=1,
                        keepdims=False
                    )

            if self.inference:
                return

            # Optimization stage
            # _target is the domain
            loss = None
            _epsilon = tf.constant(math.pow(10, -10))
            self.score_n = []
            for _target in range(self.num_domains):
                # print('_target', _target)
                ctxt_list = list(range(self.num_domains))
                ctxt_list.remove(_target)
                # print(ctxt_list)

                e = tf.gather(
                    self.joint_emb_op,
                    indices=ctxt_list,
                    axis=1
                )

                # U is the mean of the embeddings of the context
                # U is the context vector
                U = tf.reduce_mean(e, axis=1)
                # print(U.shape)

                score_pos = tf.reduce_sum(
                    tf.multiply(
                        emb_op_pos[_target],
                        U
                    ),
                    1,
                    keepdims=True
                )
                # print(score_pos.shape)
                score_pos = -score_pos

                # calculate scores for each of the rest of possible entities
                # since weights in an iteration remain same,calculate this per domain once
                # setting output to different 1s for that domain

                domain_size = self.domain_dims[_target]
                # print('Domain size', domain_size)
                domain_ids = tf.constant(np.array(list(range(domain_size))))
                domain_ids = tf.reshape(domain_ids, [-1, 1])
                # print(domain_ids)

                prev = None
                for l in range(self.num_emb_layers):
                    # print("----> Layer", (l + 1))
                    if l == 0:
                        a = tf.nn.embedding_lookup(
                            self.W[l][_target],
                            domain_ids
                        )
                        _wx = tf.squeeze(a, axis=1)
                    else:
                        _x = prev
                        _wx = tf.matmul(
                            _x,
                            self.W[l][_target]
                        )
                    if self.use_bias:
                        _wx_b = tf.add(
                            _wx,
                            self.b[l][_target]
                        )
                    else:
                        _wx_b = _wx

                    # In intermediate layers use activation
                    # if l < self.num_emb_layers - 1 and self.use_activation:
                    #     _wx_b = tf.sigmoid(_wx_b)
                    prev = _wx_b
                wx_b = prev
                # print(wx_b.shape)
                _score_n = tf.matmul(
                    U,
                    tf.transpose(wx_b)
                )

                _score_n = tf.exp(_score_n)
                # print(_score_n.shape)
                _score_n = tf.reduce_sum(
                    _score_n,
                    axis=1,
                    keepdims=True
                )
                # print(_score_n.shape)

                _score_n = tf.add(
                    _score_n,
                    _epsilon
                )

                # self.score_n.append(_score_n)
                self.score_n.append([
                    tf.constant(_target),
                    tf.reduce_min(tf.log(_score_n)),
                    tf.reduce_min(_score_n)
                ])

                _loss = tf.add(score_pos, tf.log(_score_n))
                if loss is None:
                    loss = _loss
                else:
                    loss = tf.add(loss, _loss)

            self.loss = loss
            # print('Loss shape', self.loss.shape)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )

            # self.train_opt = self.optimizer.minimize(self.loss)
            gvs = self.optimizer.compute_gradients(self.loss)
            # print([(grad, var) for grad, var in gvs])
            capped_gvs = [
                (tf.clip_by_value(grad, -1.0, 1.0), var)
                for grad, var in gvs
            ]
            self.train_opt = self.optimizer.apply_gradients(capped_gvs)
            return

    def set_pretrained_model_file(self, f_path):
        self.frozen_file = f_path
        return

    def train_model(self, x):
        print('Start of training :: ')
        self.ts = str(time.time()).split('.')[0]
        if self.test_SerialID is None:
            f_name = 'frozen' + '_' + self.model_signature + '_' + self.ts + '.pb'
        else:
            f_name = 'frozen' + '_' + self.model_signature + '_serialID_' + str(self.test_SerialID) + '_' + self.ts + '.pb'


        self.frozen_file = os.path.join(
            self.save_dir, 'checkpoints', f_name
        )

        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()
        bs = self.batch_size
        x_pos = x

        num_batches = x_pos.shape[0] // bs
        losses = []

        frozen_graph_def = None
        Check_Save_Prev = False

        print('Num batches :', num_batches)
        last_10_epochs_loss = []
        last_10_graph_defs = []

        for e in range(self.num_epochs):

            t1 = time.time()
            for _b in range(num_batches):

                _x_pos = x_pos[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x_pos = x_pos[_b * bs:]
                if _b == 0:
                    print(_x_pos.shape)

                _, sn, loss = self.sess.run(
                    [self.train_opt, self.score_n, self.loss],
                    feed_dict={
                        self.x_pos_inp: _x_pos,
                    }
                )
                losses.append(np.mean(loss))
                batch_loss = np.mean(loss)
                if _b % 200 == 0:
                    print(batch_loss)

                if np.isnan(batch_loss):
                    Check_Save_Prev = True
                    print('[ERROR] Loss is NaN !!!, breaking...')
                    break
                # else:
                #     last_10_epochs_loss.append(batch_loss)
                #     last_10_epochs_loss = last_10_epochs_loss[-10:]

                    # graph_def = tf.get_default_graph().as_graph_def()
                    # frozen_graph_def = convert_variables_to_constants(
                    #     self.sess,
                    #     graph_def,
                    #     self.wb_names
                    # )
                    # print(type(graph_def))
                    # last_10_graph_defs.append(frozen_graph_def)
                    # last_10_graph_defs = last_10_graph_defs[-10:]


            if Check_Save_Prev is True:
                break
            else:
                graph_def = tf.get_default_graph().as_graph_def()
                frozen_graph_def = convert_variables_to_constants(
                    self.sess,
                    graph_def,
                    self.wb_names
                )
                with tf.gfile.GFile(self.frozen_file, "wb") as f:
                    f.write(frozen_graph_def.SerializeToString())

                t2 = time.time()
                t = (t2 - t1) / 60
                print('Epoch ', e + 1, 'Time elapsed in epoch : ', t, 'minutes')

        # last_5_epochs_loss = last_10_epochs_loss[-5:]
        # last_5_graph_defs = last_10_graph_defs[-5:]
        # min_idx = np.argmin(last_5_epochs_loss)


        # frozen_graph_def = last_5_graph_defs[min_idx]
        # print(' > ', len(last_5_graph_defs) )
        # with tf.gfile.GFile(self.frozen_file, "wb") as f:
        #     f.write(frozen_graph_def.SerializeToString())


        # print('Losses :', losses)
        if self.save_loss_fig or self.show_loss_figure:
            plt.figure()
            plt.title('Training Loss')
            plt.xlabel('batch')
            plt.ylabel('loss')
            plt.plot(range(len(losses)), losses, 'r-')

            if self.save_loss_figure:
                fig_name = 'loss_' + self.model_signature + '_epochs_' + str(self.num_epochs) + '_' + self.ts + '.png'
                file_path = os.path.join(self.op_dir, fig_name)
                plt.savefig(file_path)

            if self.show_loss_figure:
                plt.show()

            plt.close()

        # graph_def = tf.get_default_graph().as_graph_def()
        # frozen_graph_def = convert_variables_to_constants(
        #     self.sess,
        #     graph_def,
        #     self.wb_names
        # )

        # select the one with least loss


        return self.frozen_file

    # This is an external function
    # x is the index data
    # ep is entity probability
    def get_embedding_mean(self, x):
        self.set_w_mean = False
        self.restore_model()
        output = []
        bs = self.batch_size
        num_batches = x.shape[0] // bs

        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches):
                _x = x[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x = x[_b * bs:]

                _output = sess.run(
                    self.mean_emb_op,
                    feed_dict={
                        self.x_pos_inp: _x
                    }
                )
                output.extend(_output)
            res = np.array(output)

        return res

    def get_w_embedding_mean(self, x, ep):
        self.set_w_mean = True
        self.restore_model()
        output = []
        bs = self.batch_size
        num_batches = x.shape[0] // bs

        with tf.Session(graph=self.restore_graph) as sess:
            for _b in range(num_batches):
                _x = x[_b * bs: (_b + 1) * bs]
                _ep = ep[_b * bs: (_b + 1) * bs]
                if _b == num_batches - 1:
                    _x = x[_b * bs:]
                    _ep = ep[_b * bs:]
                _output = sess.run(
                    self.w_mean_emb_op,
                    feed_dict={
                        self.x_pos_inp: _x,
                        self.entity_prob_x: _ep
                    }
                )
                output.extend(_output)
            res = np.array(output)

        # remove the 1st singular vector

        A = np.transpose(res)
        U, S, Vt = randomized_svd(A, n_components=1)
        # U, s, VT = svd(A,full_matrices=False)
        U_1 = U
        tmp = np.dot(U_1, np.transpose(U_1))

        R = []
        for r in res:
            r = r - np.matmul(tmp, r)
            R.append(r)
        R = np.array(R)

        return R

