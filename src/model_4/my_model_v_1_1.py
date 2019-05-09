# --------------------------------- #
# Calculate  record embeddings based on Arora Paper
# --------------------------------- #


import operator
import pickle
import math
import tensorflow as tf
import numpy as np
import glob
import pandas as pd
import os
from sklearn.manifold import TSNE
import sys
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import time
import inspect
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
import yaml
from sklearn.metrics import auc

sys.path.append('./..')
sys.path.append('./../../.')
try:
    from .src.Eval import evaluation_v1
except:
    from src.Eval import evaluation_v1

try :
    from .src.model_3 import lof_1
except:
    from src.model_3 import lof_1


# ------------------------------------ #
cur_path = '/'.join(
    os.path.abspath(
        inspect.stack()[0][1]
    ).split('/')[:-1]
)

sys.path.append(cur_path)
FLAGS = tf.app.flags.FLAGS

_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "3.0"
__processor__ = 'embedding'
_SAVE_DIR = 'save_dir'
MODEL_NAME = 'model_4_v1'
_DIR = None
DATA_DIR = None
MODEL_OP_FILE_PATH = None
CONFIG_FILE = 'model_4_config.yaml'

def create_args():

    tf.app.flags.DEFINE_string(
        'directory',
        'data_3',
        "path to data in the filtered/processed data")

    tf.app.flags.DEFINE_float(
        'learning_rate',
        0.00025,
        'Initial learning rate.'
    )
    tf.app.flags.DEFINE_integer(
        'batchsize',
        256,
        'size of batch'
    )
    tf.app.flags.DEFINE_string(
        'op_dims',
        '8,4',
        'size of embedding'
    )

    tf.app.flags.DEFINE_integer(
        'num_epochs',
        100,
        'number of epochs for training'
    )

    tf.app.flags.DEFINE_string(
        'saved_model_file',
        None,
        "path to data"
    )

    tf.app.flags.DEFINE_boolean(
        'use_pretrained',
        False,
        "To train a new model or use pre trained model"
    )

    tf.app.flags.DEFINE_integer(
        'run_case',
        0,
        'run case for getting embedding of the transactions'
    )

    tf.app.flags.DEFINE_integer(
        'viz_case',
        0,
        'visualization'
    )

    tf.app.flags.DEFINE_boolean(
        'score',
        False,
        'calculate scores'
    )

    tf.app.flags.DEFINE_boolean(
        'show_loss_figure',
        True,
        'set display of figure (based on matplotlib backend support)'
    )

    tf.app.flags.DEFINE_boolean(
        'save_loss_figure',
        True,
        'set display of figure based on matplotlib backend support'
    )

    tf.app.flags.DEFINE_boolean(
        'use_activation',
        False,
        'use activation function'
    )

    tf.app.flags.DEFINE_boolean(
        'use_bias',
        True,
        'use bias'
    )
    tf.app.flags.DEFINE_boolean(
        'use_drop_out',
        True,
        'use bias'
    )

    return


def get_domain_dims():
    global DATA_DIR
    f_path = os.path.join(DATA_DIR, 'domain_dims.pkl')
    with open(f_path, 'rb') as fh:
        res = pickle.load(fh)
    print(res)
    return list(res.values())

# ----------------------------------------- #
# ---------		  Model Config	  --------- #
# ----------------------------------------- #

embedding_dims = None
DOMAIN_DIMS = None

def setup_general_config():
    global MODEL_NAME
    global _DIR
    global SAVE_DIR
    global OP_DIR
    global _SAVE_DIR
    global config

    SAVE_DIR = os.path.join(config['SAVE_DIR'], _DIR)
    OP_DIR = os.path.join(config['OP_DIR'], _DIR)
    print(cur_path)
    print(OP_DIR)

    if not os.path.exists(config['OP_DIR']):
        os.mkdir(config['OP_DIR'])

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    if not os.path.exists(config['SAVE_DIR']):
        os.mkdir(os.path.join(config['SAVE_DIR']))

    if not os.path.exists(SAVE_DIR):
        os.mkdir(os.path.join(SAVE_DIR))
    return



class model:
    def __init__(self):
        global SAVE_DIR
        global OP_DIR
        self.inference = False
        self.save_dir = SAVE_DIR
        self.op_dir = OP_DIR
        self.frozen_file = None
        self.ts = None
        self.use_bias = False
        self.save_loss_fig = True
        self.show_figure = False
        self.use_activation = False

        return

    def set_model_hyperparams(
            self,
            domain_dims,
            emb_dims,
            use_acivation=False,
            use_bias=False,
            batch_size=128,
            num_epochs=20,
            learning_rate=0.001
    ):
        global MODEL_NAME
        self.learning_rate = learning_rate
        self.num_domains = len(domain_dims)
        self.domain_dims = domain_dims
        self.num_emb_layers = len(emb_dims)
        self.emb_dims = emb_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model_signature = MODEL_NAME + '_'.join([str(e) for e in emb_dims])
        self.use_activation = use_acivation
        self.use_bias = use_bias
        self.alpha = 0.1
        return

    def set_model_options(
            self,
            show_loss_figure,
            save_loss_figure
    ):
        self.show_loss_figure = show_loss_figure
        self.save_loss_figure = save_loss_figure
        return

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
        print('>> Defining weights :: start')

        self.W = [None] * self.num_emb_layers
        self.b = [None] * self.num_emb_layers

        wb_scope_name = 'params'
        # doman dimensions

        layer_1_dims = []
        for i in self.domain_dims:
            _d = int(math.ceil(math.log(i, 2)))
            if _d <= 1 :
                _d += 1
            layer_1_dims.append(_d)
        print(layer_1_dims)

        with tf.name_scope(wb_scope_name):
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            self.wb_names = []

            # -------
            # For each layer define W , b
            # -------
            for l in range(self.num_emb_layers):
                self.W[l] = [None] * self.num_domains
                self.b[l] = [None] * self.num_domains

                print("----> Layer", (l + 1))
                if l == 0:
                    layer_inp_dims = self.domain_dims
                    # layer_op_dims = [self.emb_dims[l]] * self.num_domains
                    layer_op_dims = layer_1_dims
                    print(layer_inp_dims)
                    print(layer_op_dims)
                else:
                    if l == 1:
                        layer_inp_dims = layer_1_dims
                    else:
                        layer_inp_dims = [self.emb_dims[l - 1]] * self.num_domains
                    layer_op_dims = [self.emb_dims[l]] * self.num_domains
                    print(layer_inp_dims)
                    print(layer_op_dims)
                for d in range(self.num_domains):
                    print('-> Domain', (d + 1))
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
        print(self.wb_names)
        print('>> Defining weights :: end')

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
        print('ts ::', self.ts)

        tf.reset_default_graph()
        print('Frozen file', self.frozen_file)

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
        print('Building model : start ')
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
                    print("----> Layer", (l + 1))
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
                print(x_pos_WXb[d].shape)

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
                self.entity_prob_x = tf.placeholder(
                    tf.float32,
                    [None, self.num_domains],
                    name='entity_prob_x'
                )

                _ep_x = tf.add(self.entity_prob_x, self.alpha)
                self.ep_x = tf.add( tf.pow(_ep_x,-1), self.alpha)

                # emb_op_pos : shape [ batch, domains, emd_dim]
                # entity_prob_x : shape [batch, dommains]
                ep_x = tf.reshape(self.ep_x,[-1,self.ep_x.shape[-1],1])
                ep_x = tf.tile(ep_x, [1, 1, self.emb_dims[-1]])
                print('ep_x shape' , ep_x.shape)

                # apply  weight : element wise multiply
                w_joint_emb = tf.math.multiply(self.joint_emb_op,ep_x)

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
                print('_target', _target)
                ctxt_list = list(range(self.num_domains))
                ctxt_list.remove(_target)
                print(ctxt_list)

                e = tf.gather(
                    self.joint_emb_op,
                    indices=ctxt_list,
                    axis=1
                )

                # U is the mean of the embeddings of the context
                # U is the context vector
                U = tf.reduce_mean(e, axis=1)
                print(U.shape)

                score_pos = tf.reduce_sum(
                    tf.multiply(
                        emb_op_pos[_target],
                        U
                    ),
                    1,
                    keepdims=True
                )
                print(score_pos.shape)
                score_pos = -score_pos

                # calculate scores for each of the rest of possible entities
                # since weights in an iteration remain same,calculate this per domain once
                # setting output to different 1s for that domain

                domain_size = self.domain_dims[_target]
                print('Domain size', domain_size)
                domain_ids = tf.constant(np.array(list(range(domain_size))))
                domain_ids = tf.reshape(domain_ids, [-1, 1])
                print(domain_ids)

                prev = None
                for l in range(self.num_emb_layers):
                    print("----> Layer", (l + 1))
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
                print(wx_b.shape)
                _score_n = tf.matmul(
                    U,
                    tf.transpose(wx_b)
                )

                _score_n = tf.exp(_score_n)
                print(_score_n.shape)
                _score_n = tf.reduce_sum(
                    _score_n,
                    axis=1,
                    keepdims=True
                )
                print(_score_n.shape)

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
            print('Loss shape', self.loss.shape)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            )

            # self.train_opt = self.optimizer.minimize(self.loss)
            gvs = self.optimizer.compute_gradients(self.loss)
            print([(grad, var) for grad, var in gvs])
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

        f_name = 'frozen' + '_' + self.model_signature + '_' + self.ts + '.pb'

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
        print('Num batches :', num_batches)
        for e in range(self.num_epochs):
            t1 = time.time()
            for _b in range(num_batches):

                _x_pos = x_pos[_b * bs: (_b + 1) * bs]

                # from pprint import pprint
                # pprint(_x_pos)
                # print(_x_pos.shape)
                # print(get_domain_dims())
                # exit(1)
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
                print(np.mean(loss))
                # print(sn)
                # print('-----------')
                if np.isnan(np.mean(loss)):
                    break

            t2 = time.time()
            t = (t2 - t1) / 60
            print('Epoch ', e + 1, 'Time elapsed in epoch : ', t, 'minutes')

        print('Losses :', losses)
        if self.save_loss_fig or self.show_loss_figure:
            plt.figure()
            plt.title('Training Loss')
            plt.xlabel('batch')
            plt.ylabel('loss')
            plt.plot(range(len(losses)), losses, 'r-')
            if self.show_loss_figure:
                plt.show()
            if self.save_loss_figure:
                fig_name = 'loss_'+ self.model_signature + '_epochs_' +str(self.num_epochs) +'_' + self.ts + '.png'
                file_path = os.path.join(self.op_dir, fig_name)
                plt.savefig(file_path)
            plt.close()

        graph_def = tf.get_default_graph().as_graph_def()
        frozen_graph_def = convert_variables_to_constants(
            self.sess,
            graph_def,
            self.wb_names
        )

        with tf.gfile.GFile(self.frozen_file, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())
        return

    # This is an external function
    # x is the index data
    # ep is entity probability
    def get_embedding_mean(self, x):

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
                    self.w_mean_emb_op,
                    feed_dict={
                        self.x_pos_inp: _x
                    }
                )
                output.extend(_output)
            res = np.array(output)

        return res

    def get_w_embedding_mean(self, x, ep):

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
        from scipy.linalg import svd
        from sklearn.decomposition import TruncatedSVD
        from sklearn.utils.extmath import randomized_svd


        A = np.transpose(res)
        U, S, Vt = randomized_svd(A, n_components=1)
        # U, s, VT = svd(A,full_matrices=False)
        U_1 = U
        tmp = np.dot(U_1, np.transpose(U_1))

        R = []
        for r in res:
            r = r - np.matmul(tmp,r)
            R.append(r)
        R = np.array(R)

        return R


# --------------------------------------------- #

def set_up_model():
    global embedding_dims

    if FLAGS.op_dims is not None:
        embedding_dims = [int(_) for _ in FLAGS.op_dims.split(',')]
    else:
        embedding_dims = embedding_dims

    print(embedding_dims)

    model_obj = model()
    model_obj.set_model_options(
        show_loss_figure = FLAGS.show_loss_figure,
        save_loss_figure = FLAGS.save_loss_figure
    )

    domain_dims = get_domain_dims()

    model_obj.set_model_hyperparams(
        domain_dims=domain_dims,
        emb_dims=embedding_dims,
        batch_size=FLAGS.batchsize,
        num_epochs=FLAGS.num_epochs,
        learning_rate=FLAGS.learning_rate,
        use_acivation=FLAGS.use_activation
    )
    model_obj.build_model()
    return model_obj


def get_data():
    global DATA_FILE
    global _DIR
    global DATA_DIR

    DATA_FILE = os.path.join(DATA_DIR, 'train_x.pkl')

    with open(DATA_FILE, 'rb') as fh:
        DATA_X = pickle.load(fh)
    print(DATA_X.shape)

    _test_files = os.path.join(
        DATA_DIR,
        'test_x_*.pkl'
    )
    print(_test_files)
    test_files = glob.glob(_test_files)
    test_x = []
    test_anom_id = []
    test_all_id = []
    for t in test_files:
        with open(t, 'rb') as fh:
            data = pickle.load(fh)
            test_anom_id.append(data[0])
            test_all_id.append(data[1])
            test_x.append(data[2])

    train_ids = None
    train_id_file = os.path.join(DATA_DIR, 'train_x_id.pkl')
    with open(train_id_file, 'rb') as fh:
        train_ids = pickle.load(fh)

    entity_prob_train_file = os.path.join(DATA_DIR, 'entity_prob_train_x.pkl')

    with open(entity_prob_train_file, 'rb') as fh:
        entity_prob_train_x = pickle.load(fh)

    _entity_prob_test_files =  os.path.join(
        DATA_DIR,
        'entity_prob_test_x_*.pkl'
    )

    entity_prob_test_files = glob.glob(_entity_prob_test_files)
    entity_prob_test = []

    for t in entity_prob_test_files:
        with open(t, 'rb') as fh:
            data = pickle.load(fh)
            entity_prob_test.append(data)

    return DATA_X, test_anom_id, test_all_id, test_x , train_ids, entity_prob_train_x, entity_prob_test


def main(argv=None):
    global embedding_dims
    global SAVE_DIR
    global _DIR
    global DATA_DIR
    global config
    global CONFIG_FILE
    global MODEL_NAME
    global DOMAIN_DIMS

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    _DIR = config['_DIR']
    DATA_DIR = config['DATA_DIR'] + '/' + _DIR
    setup_general_config()

    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(os.path.join(SAVE_DIR, 'checkpoints'))

    # ------------ #

    data_x, test_anom_id, test_all_id, test_x, train_ids , entity_prob_train_x , entity_prob_test = get_data()
    DOMAIN_DIMS = get_domain_dims()

    model_obj = set_up_model()

    _use_pretrained = FLAGS.use_pretrained
    if _use_pretrained is False:
        model_obj.train_model(data_x)


    if _use_pretrained is True:
        pretrained_file = None
        if config['saved_model_file'] is None:
            if FLAGS.saved_model_file is not None:
                pretrained_file = FLAGS.saved_model_file
        else:
            pretrained_file = config['saved_model_file']
        print('Pretrained File :', pretrained_file)

        print('Saved file ::', FLAGS.saved_model_file)
        saved_file_path = os.path.join(
            SAVE_DIR,
            'checkpoints',
            pretrained_file
        )
        model_obj.set_pretrained_model_file(saved_file_path)


    test_result_r = []
    test_result_p = []

    for i in range(len(test_x)):

        # combine the test and train data - since it is a density based method
        _x = np.vstack([data_x, test_x[i]])
        _ep = np.vstack([entity_prob_train_x, entity_prob_test[i]])

        mean_embeddings = model_obj.get_w_embedding_mean(_x,_ep)
        print(data_x.shape[0], test_x[i].shape[0], _x.shape[0] ,mean_embeddings.shape[0])

        _test_all_id = test_all_id[i]


        _all_ids = list(train_ids)
        _all_ids.extend(list(_test_all_id))


        anomalies = test_anom_id[i]


        # USE LOF here
        sorted_id_score_dict = lof_1.anomaly_1(
            id_list=_all_ids,
            embed_list=mean_embeddings
        )
        print(' >>>> ', len(sorted_id_score_dict))

        _scored_dict_test = {}

        for k1,v in sorted_id_score_dict.items():
            if k1 in _test_all_id or k1 in _scored_dict_test:
                _scored_dict_test[k1] = v

        recall, precison = evaluation_v1.precision_recall_curve(
            _scored_dict_test,
            anomaly_id_list=anomalies
        )
        test_result_r.append(recall)
        test_result_p.append(precison)

        print('--------------------------')
        # _auc = auc(recall, precison)
        # plt.figure(figsize=[14, 8])
        # plt.plot(
        #     recall,
        #     precison,
        #     color='blue', linewidth=1.75)
        # plt.xlabel('Recall', fontsize=15)
        # plt.ylabel('Precision', fontsize=15)
        # plt.title('Recall | AUC ' + str(_auc), fontsize=15)
        # f_name = 'precison-recall_1_test_' + str(i) + '.png'
        # f_path = os.path.join(OP_DIR, f_name)
        # # plt.savefig(f_path)
        # plt.close()


    plt.figure(figsize=[14, 8])
    j = 1
    res_str = 'auPR : '
    for _x,_y in zip(test_result_r,test_result_p):
        plt.plot(
            _x,
            _y,
            linewidth=1.75,
            label='Test set ' + str(j)
        )
        j += 1
        _auc = auc(_x, _y)
        res_str += ' ' + "{0:.2f}".format(_auc)
        print(_auc)

    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.title('Precision Recall Curve ' + res_str, fontsize=18)
    plt.legend(loc='best')
    f_name = 'precison-recall_test_' + str(i) + '.png'
    f_path = os.path.join(OP_DIR, f_name)
    plt.savefig(f_path)
    plt.show()
    plt.close()


def internal_main(argv):
    main(argv)
    return


def external_main():
    return main(None)

# ----------------------------------------------------------------- #
if __name__ == "__main__":
    create_args()
    tf.app.run(internal_main)
elif __name__ == "my_model_v_1_0" :
    create_args()
    external_main()

# ----------------------------------------------------------------- #
