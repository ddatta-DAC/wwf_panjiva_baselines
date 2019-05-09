# ----------------------- #
# A modified version of APE
# with SGNS instead
# ----------------------- #
import operator
import pickle
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import inspect
import matplotlib.pyplot as plt
import sys
import yaml
import time
from collections import OrderedDict
import seaborn as sns
import glob

sys.path.append('./..')
sys.path.append('./../../.')
try:
    from .src.Eval import evaluation_v1
except:
    from src.Eval import evaluation_v1

cur_path = '/'.join(
    os.path.abspath(
        inspect.stack()[0][1]
    ).split('/')[:-2]
)
sys.path.append(cur_path)
FLAGS = tf.app.flags.FLAGS

# ------------------------- #

# ------------------------- #
_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "6.0"


# ------------------------- #


def create_args():
    tf.app.flags.DEFINE_string(
        'dir',
        'None',
        "path to data")

    tf.app.flags.DEFINE_float(
        'learning_rate',
        0.001,
        'Initial learning rate.'
    )
    tf.app.flags.DEFINE_integer(
        'batchsize',
        1024,
        'size of batch'
    )
    tf.app.flags.DEFINE_integer(
        'op_dim',
        8,
        'size of embedding'
    )

    tf.app.flags.DEFINE_integer(
        'num_epochs',
        2,
        'number of epochs for training'
    )

    tf.app.flags.DEFINE_integer(
        'neg_samples',
        3,
        'number of neg samples for training'
    )

    tf.app.flags.DEFINE_boolean(
        'show_loss_fig',
        False,
        'Display figure'
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
        2,
        'run case for getting embedding of the transactions'
    )

    tf.app.flags.DEFINE_integer(
        'viz',
        3,
        'visualization'
    )

    tf.app.flags.DEFINE_boolean(
        'score',
        False,
        'calculate scores'
    )

    return


def get_domain_arity():
    global DATA_DIR
    global _DIR
    f = os.path.join(DATA_DIR, _DIR, 'domain_dims.pkl')

    with open(f, 'rb') as fh:
        dd = pickle.load(fh)

    return list(dd.values())


def get_cur_path():
    this_file_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

    os.chdir(this_file_path)
    print(os.getcwd())
    return this_file_path


# -------- Globals --------- #

# ---------		  Model Config	  --------- #


CONFIG_FILE = 'config_1.yaml'
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

SAVE_DIR = config['SAVE_DIR']
_DIR = config['_DIR']
OP_DIR = config['OP_DIR']
DATA_DIR = config['DATA_DIR']
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
SAVE_DIR = os.path.join(SAVE_DIR, _DIR)

print(OP_DIR)
print(SAVE_DIR)
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if not os.path.exists(OP_DIR):
    os.mkdir(OP_DIR)

OP_DIR = os.path.join(OP_DIR, _DIR)

if not os.path.exists(OP_DIR):
    os.mkdir(OP_DIR)
MODEL_NAME = 'model_ape'

embedding_dims = [16]
domain_dims = get_domain_arity()
cur_path = get_cur_path()

print(cur_path)


# ----------------------------------------- #

class model_ape_1:
    def __init__(self):
        return

    def set_hyper_parameters(
            self,
            emb_dims,
            use_bias
    ):
        self.emb_dims = emb_dims
        self.num_layers = len(emb_dims)
        self.use_bias = use_bias
        self.epsilon = 0.000001

        self.emb_str = '_'.join([str(_) for _ in self.emb_dims])
        f_name = MODEL_NAME + '_' + self.emb_str + '_k_' + str(self.neg_samples) +"_frozen.pb"
        self.frozen_filename = os.path.join(self.chkpt_dir, f_name)
        print(self.frozen_filename)

    def set_model_params(
            self,
            num_entities,
            neg_samples,
            inp_dims,
            batch_size=32,
            num_epochs=5,
            chkpt_dir=None
    ):
        global MODEL_NAME
        self.neg_samples = neg_samples  # k in paper
        self.k = neg_samples
        self.num_entities = num_entities
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.inp_dims = inp_dims
        self.chkpt_dir = chkpt_dir

    # -------- Restore ---------- #
    def restore_model(self):
        tf.reset_default_graph()

        with tf.gfile.GFile(self.frozen_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        self.restore_graph = None

        with tf.Graph().as_default() as g:
            tf.graph_util.import_graph_def(
                graph_def,
                input_map=None,
                name='',
                return_elements=None,
                op_dict=None,
                producer_op_list=None
            )
            self.restore_graph = g
            self.build_model(
                inference=True
            )
        return

    # ------- Define weights ------ #
    def get_weight_variable(self, shape, name=None):
        initial = tf.random.uniform(shape, -0.10, 0.10)
        if name is not None:
            return tf.Variable(
                initial,
                name=name
            )
        else:
            return tf.Variable(
                initial
            )

    def define_wbs(self):
        print('Defining weights :: start')
        self.W = [None] * self.num_layers
        self.B = [None] * self.num_layers

        for l in range(self.num_layers):
            W_l = [None] * self.num_entities
            self.W[l] = W_l

            # Only if bias is being used in that layer
            if self.use_bias[l] is True:
                B_l = [None] * self.num_entities
                self.B[l] = B_l

        # store names to restore graph
        self.wb_names = []

        wb_scope_name = 'params'
        _inp_dim = None
        self.W = [None] * self.num_layers
        self.B = [None] * self.num_layers

        with tf.name_scope(wb_scope_name):
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            # Declare/Define weights
            for l in range(self.num_layers):
                self.W[l] = [None] * self.num_entities
                self.B[l] = [None] * self.num_entities

                for i in range(self.num_entities):
                    if l == 0:
                        _inp_dim = self.inp_dims[i]
                    else:
                        _inp_dim = self.emb_dims[l - 1]

                    w_dims = [_inp_dim, self.emb_dims[l]]
                    name = 'W_' + str(l) + str(i)
                    if self.restore is True:
                        n = prefix + name + ':0'
                        self.W[l][i] = self.restore_graph.get_tensor_by_name(n)
                    else:
                        z = self.get_weight_variable(w_dims, name)
                        self.W[l][i] = z
                    self.wb_names.append(prefix + name)

                    # Only if bias is being used for the layer #
                    if self.use_bias[l] is True:
                        name = 'B_' + str(l) + str(i)
                        b_dims = [self.emb_dims[l]]
                        if self.restore is True:
                            n = prefix + name + ':0'
                            self.B[l][i] = self.restore_graph.get_tensor_by_name(n)
                        else:
                            z = self.get_weight_variable(b_dims, name)
                            self.B[l][i] = z
                        self.wb_names.append(prefix + name)

            print('------')

            name = 'c'
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            if self.restore is False:
                self.c = tf.Variable(
                    initial_value=tf.random.uniform([1], 0, 1),
                    name=name
                )
                self.wb_names.append(prefix + name)
            else:
                n = prefix + name + ':0'
                self.c = self.restore_graph.get_tensor_by_name(n)
            print(self.c)

            name = 'W_ij'
            prefix = self.model_scope_name + '/' + wb_scope_name + '/'
            if self.restore is False:
                self.W_ij = tf.Variable(
                    initial_value=tf.random.uniform(
                        [self.num_entities, self.num_entities],
                        0,
                        1),
                    name=name
                )
                self.wb_names.append(prefix + name)
            else:
                n = prefix + name + ':0'
                self.W_ij = self.restore_graph.get_tensor_by_name(n)

            print(self.W_ij)
            print('Defining weights :: end')

        return

    # inference = true -> inference stage, Use pre-trained model
    # inference_case valid only for inference=True
    # Note this is called after restoring model as well

    def build_model(
            self,
            inference=False
    ):
        # Set placeholders

        self.restore = inference
        # Input indices
        print('Building model : start ')
        self.output_node_names = []
        self.model_scope_name = 'model'

        with tf.variable_scope(self.model_scope_name):
            self.x_pos_inp = tf.placeholder(
                tf.int32,
                [None, self.num_entities],
                name='x_pos_ev'
            )

            # -------------------- #
            # The model already trained , no need for negative samples
            if inference is False:
                # Negative sample
                self.x_neg_inp = tf.placeholder(
                    tf.int32,
                    [None, self.neg_samples, self.num_entities],
                    name='x_neg_samples'
                )

                # Placeholder for probabilities of each entity in each domain
                # Each entity_domain has own list
                self.term_2 = tf.placeholder(
                    tf.float32,
                    shape=([None, 1])
                )

                self.term_4 = tf.placeholder(
                    tf.float32,
                    shape=([None, self.neg_samples])
                )

            # -------------------- #

            self.define_wbs()

            x_pos_inp = tf.split(
                self.x_pos_inp,
                self.num_entities,
                axis=-1
            )

            for i in range(self.num_entities):
                z_i = x_pos_inp[i]
                z_i = tf.one_hot(
                    axis=-1,
                    indices=z_i,
                    depth=self.inp_dims[i]
                )
                z_i = tf.squeeze(z_i, axis=1)
                x_pos_inp[i] = z_i

            print('Shape of inputs , after one hot encoding')
            print([k.shape.as_list() for k in x_pos_inp])

            if inference is False:
                x_neg_inp = tf.split(
                    self.x_neg_inp,
                    self.num_entities,
                    axis=-1
                )

                for i in range(self.num_entities):
                    z_i = x_neg_inp[i]
                    z_i = tf.one_hot(
                        axis=-1,
                        indices=z_i,
                        depth=self.inp_dims[i]
                    )
                    z_i = tf.squeeze(z_i, axis=2)
                    x_neg_inp[i] = z_i

                print('Shape of inputs , after one hot encoding')
                print(
                    [k.shape.as_list() for k in x_neg_inp]
                )
            print('-----')

            x_pos_WXb = [None] * self.num_layers

            # WX + B
            # for positive samples
            for l in range(0, 1):
                x_pos_WXb[l] = [None] * self.num_entities
                for i in range(self.num_entities):
                    if l == 0:
                        _x = x_pos_inp[i]
                    else:
                        _x = x_pos_WXb[l - 1][i]
                    _wx = tf.matmul(_x, self.W[l][i])
                    if self.use_bias[l]:
                        _wx_b = tf.add(_wx, self.B[l][i])
                    else:
                        _wx_b = _wx
                    print(_wx_b)
                    x_pos_WXb[l][i] = _wx_b

            # ---------------- #
            # WX + B
            # for negative samples

            if inference is False:
                x_neg_WXb = [None] * self.num_layers
                for l in range(0, 1):
                    x_neg_WXb[l] = [None] * self.num_entities
                    for i in range(self.num_entities):
                        if l == 0:
                            _x = x_neg_inp[i]
                        else:
                            _x = x_neg_WXb[l - 1][i]
                        print(self.W[l][i])
                        _wx = tf.einsum('ijk,kl -> ijl', _x, self.W[l][i])

                        if self.use_bias[l]:
                            _wx_b = tf.add(_wx, self.B[l][i])
                        else:
                            _wx_b = _wx
                        x_neg_WXb[l][i] = _wx_b
            # ---------------- #
            # x_pos_WXb has the embedding for the different entities for positive samples
            # do pairwise dot product

            _sum = 0
            for i in range(self.num_entities):
                for j in range(i + 1, self.num_entities):
                    z = tf.reduce_sum(
                        tf.multiply(x_pos_WXb[l][i],
                                    x_pos_WXb[l][j]),
                        axis=1,
                        keepdims=True) * tf.sqrt(tf.square(self.W_ij[i][j]))
                    _sum += z
                    print(z)
            P_e = tf.exp(_sum + self.c)  # 1st term in the loss equation
            self.score = P_e

            # --------------------------------- #
            # End of build	-> for restore mode
            # --------------------------------- #
            if inference is True:
                return

            # ---------------- #
            # x_pos_WXb has the embedding for the different entities for negative samples
            # do pairwise dot product
            neg_pair_dp = []
            for i in range(self.num_entities):
                for j in range(i + 1, self.num_entities):
                    print(x_neg_WXb[l][i])
                    _z1 = _wx = tf.einsum(
                        'ijk,ijk -> ijk',
                        x_neg_WXb[l][i],
                        x_neg_WXb[l][j]
                    )
                    z = tf.reduce_sum(
                        _z1,
                        axis=-1,
                        keepdims=True) *  tf.sqrt(tf.square(self.W_ij[i][j]))
                    neg_pair_dp.append(z)
            print(neg_pair_dp)

            z1 = tf.stack(neg_pair_dp, axis=2)
            z1 = tf.squeeze(z1, axis=-1)
            z2 = tf.reduce_sum(z1, axis=-1, keepdims=False)
            z2 = z2 + self.c
            z3 = tf.exp(z2)  # 3rd term in the loss equation

            # ---------------- #

            obj = tf.log(tf.sigmoid(tf.log(P_e) - self.term_2) + self.epsilon)
            print('1', tf.sigmoid(tf.log(P_e)))
            print('z3', z3)
            z4 = tf.log(tf.sigmoid(-z3 + self.term_4) + self.epsilon)
            z5 = tf.reduce_sum(z4, axis=-1, keepdims=True)
            obj += z5
            self.obj = obj
            print(self.obj)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.train_opt = self.optimizer.minimize(-self.obj)

    def train_model(self, data):
        global OP_DIR
        global _DIR
        self.sess = tf.InteractiveSession()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()
        bs = self.batch_size
        x_pos = data[0]
        x_neg = data[1]
        term_2 = data[2]
        term_4 = data[3]
        num_batches = x_pos.shape[0] // bs
        print('-----')
        losses = []
        for e in range(self.num_epochs):
            print('epoch', e + 1)
            st = time.time()
            for _b in range(num_batches):
                if _b % 100 == 0:
                    print('Batch :', _b)
                _x_pos = x_pos[_b * bs: (_b + 1) * bs]
                _x_neg = x_neg[_b * bs: (_b + 1) * bs]
                _term_2 = term_2[_b * bs: (_b + 1) * bs]
                _term_4 = term_4[_b * bs: (_b + 1) * bs]

                _, loss = self.sess.run(
                    [self.train_opt, self.obj],
                    feed_dict={
                        self.x_pos_inp: _x_pos,
                        self.x_neg_inp: _x_neg,
                        self.term_2: _term_2,
                        self.term_4: _term_4
                    }
                )
                if _b % 100 == 0:
                    print('Loss :', np.mean(loss))

                losses.append(np.mean(loss))

            ed = time.time()
            print('Time elapsed: ', ed - st)
        # print(np.mean(loss))

        print('------------------------------->')
        losses = np.array(losses) * -1

        fig_name = 'model_ape_1_' + str(time.time()) + '.png'
        fig_path = os.path.join(OP_DIR, fig_name)
        plt.figure(figsize=[14, 10])
        plt.title('Loss for model training', fontsize=18)
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Loss value', fontsize=16)
        sns.lineplot(x=range(len(losses)), y=losses, lw=0.75)
        plt.plot(range(len(losses)), losses, 'r-')
        plt.savefig(fig_path)
        plt.close()

        graph_def = tf.get_default_graph().as_graph_def()
        frozen_graph_def = convert_variables_to_constants(
            self.sess,
            graph_def,
            self.wb_names
        )

        with tf.gfile.GFile(self.frozen_filename, "wb") as f:
            f.write(frozen_graph_def.SerializeToString())
        return

    def inference(self,
                  data,
                  ):
        self.restore_model()
        output = None
        print(data.shape)
        with tf.Session(graph=self.restore_graph) as sess:
            output = sess.run(
                self.score,
                feed_dict={
                    self.x_pos_inp: data,
                })

        res = np.array(output)
        return res


# ---------------------------- #
# Return data for training
# x_pos_inp : [?, num_entities]
# x_neg_inp = [?,neg_samples, num_entities]

def get_training_data(
        data_x,
        neg_samples
):
    global DATA_DIR
    global _DIR
    global config

    vals = data_x

    # Calculate probability of each entity for each domain
    num_domains = data_x.shape[1]

    P_A = [None] * num_domains
    inp_dims = [None] * num_domains

    domain_dims = get_domain_arity()
    for d in range(0, num_domains):
        _col = np.reshape(data_x[:, [d]], -1)
        _series = pd.Series(_col)
        tmp = _series.value_counts(normalize=True)
        P_Aa = tmp.to_dict()
        for _z in range(domain_dims[d]):
            if _z not in P_Aa.keys():
                P_Aa[_z] = math.pow(10,-8)
        P_A[d] = P_Aa
        inp_dims[d] = domain_dims[d]
    # print('Input dimensions', inp_dims)

    TRAIN_DATA_FILE_NAME = 'ape_v1_train_data_'+ str(neg_samples) +'.pkl'
    TRAIN_DATA_FILE = os.path.join(DATA_DIR, _DIR, TRAIN_DATA_FILE_NAME)
    print(TRAIN_DATA_FILE)
    if os.path.exists(TRAIN_DATA_FILE):
        with open(TRAIN_DATA_FILE, 'rb') as fh:
            data = pickle.load(fh)
        if config['REFRESH_DATA'] is False:
            return data, inp_dims

    print(' Creating training data ')
    x_pos = []
    x_neg = []
    term_2 = []
    term_4 = []
    count = 0


    for row in vals:
        count +=1
        print('>', count)
        val = row
        for nd in range(num_domains):

            record = list(val)
            x_pos.append(record)
            cur = record[nd]
            _x_neg = []
            _term_4 = []

            for n in range(neg_samples):
                record = list(val)
                # replace
                # do a uniform sampling
                rnd = None

                while True:
                    rnd = np.random.randint(
                        low=0,
                        high=inp_dims[nd]
                    )
                    if inp_dims[nd] == 1 or rnd != cur:
                        break
                record[nd] = rnd
                _x_neg.append(record)
                _term_4.append(np.log(P_A[nd][rnd]))

            log_kPne = 0.0
            for _d2 in range(num_domains):
                log_kPne += np.log(P_A[_d2][record[_d2]])
            log_kPne /= num_domains

            x_neg.append(_x_neg)
            term_4.append(_term_4)
            term_2.append([log_kPne])

    x_pos = np.array(x_pos)
    x_neg = np.array(x_neg)
    term_2 = np.array(term_2)
    term_4 = np.array(term_4)

    print(x_pos.shape)
    print(x_neg.shape)
    print(term_2.shape)
    print(term_4.shape)

    data = [x_pos, x_neg, term_2, term_4]
    # save data

    with open(TRAIN_DATA_FILE, 'wb') as fh:
        pickle.dump(data, fh, pickle.HIGHEST_PROTOCOL)

    return data, inp_dims

# ----------------------------------------- #
# This gets all the data
# ----------------------------------------- #
def get_data():
    global DATA_FILE
    global _DIR
    global DATA_DIR

    DATA_FILE = os.path.join(DATA_DIR, _DIR, 'train_x.pkl')

    with open(DATA_FILE, 'rb') as fh:
        DATA_X = pickle.load(fh)
    print(DATA_X.shape)

    _test_files = os.path.join(
        DATA_DIR,
        _DIR,
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

    return DATA_X, test_anom_id, test_all_id, test_x



# --------------------------- #
def main(argv):
    global _DIR
    global OP_DIR
    global SAVE_DIR

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        if os.path.exists(os.path.join(SAVE_DIR, _DIR)):
            os.mkdir(os.path.join(SAVE_DIR, _DIR))

    checkpoint_dir = os.path.join(SAVE_DIR)

    print(os.getcwd())



    data_x, test_anom_id, test_all_id, test_x =  get_data()
    data, inp_dims= get_training_data(
        data_x,
        FLAGS.neg_samples
    )


    num_domains = len(inp_dims)
    model_obj = model_ape_1()
    model_obj.set_model_params(
        num_entities=num_domains,
        inp_dims=inp_dims,
        neg_samples=FLAGS.neg_samples,
        batch_size=FLAGS.batchsize,
        num_epochs=FLAGS.num_epochs,
        chkpt_dir=checkpoint_dir
    )

    model_obj.set_hyper_parameters(
        emb_dims=[8],
        use_bias=[True, False]
    )

    print(FLAGS.use_pretrained)
    if FLAGS.use_pretrained is False:
        model_obj.build_model()
        model_obj.train_model(data)

    test_result_r = []
    test_result_p = []
    res = None
    for i in range(len(test_x)-1):

        _x = test_x[i]
        res = model_obj.inference(_x)
        all_ids = test_all_id[i]
        anomalies = test_anom_id[i]

        _id_score_dict = {
            id: res for id, res in zip(all_ids, res)
        }
        tmp = sorted(
            _id_score_dict.items(),
            key=operator.itemgetter(1)
        )
        sorted_id_score_dict = OrderedDict()
        for e in tmp:
            sorted_id_score_dict[e[0]] = e[1]

        recall, precison = evaluation_v1.precision_recall_curve(
            sorted_id_score_dict,
            anomaly_id_list=anomalies
        )

        print('--------------------------')

        from sklearn.metrics import auc
        _auc = auc(recall, precison)
        plt.figure(figsize=[14, 8])
        plt.plot(
            recall,
            precison,
            color='blue', linewidth=1.75)

        plt.xlabel('Recall', fontsize=15)
        plt.ylabel('Precision', fontsize=15)
        plt.title('Recall | AUC ' + str(_auc), fontsize=15)
        f_name = 'precison-recall_1_test_' + str(i) + '.png'
        f_path = os.path.join(OP_DIR, f_name)

        # plt.savefig(f_path)
        test_result_r.append(recall)
        test_result_p.append(precison)
        plt.close()

        print('----------------------------')

        x, y = evaluation_v1.performance_by_score(
            sorted_id_score_dict,
            anomalies)

        plt.figure(figsize=[14, 8])
        plt.plot(
            x,
            y,
            color='red', linewidth=1.75)
        # plt.xlabel(' ', fontsize=15)
        plt.ylabel('Percentage of anomalies detected', fontsize=15)
        plt.title('Lowest % of scores', fontsize=15)

        f_name = 'score_1_test_' + str(i) + '.png'
        f_path = os.path.join(OP_DIR, f_name)

        plt.savefig(f_path)
        plt.close()

    plt.figure(figsize=[14, 8])
    j = 1

    for _x, _y in zip(test_result_r, test_result_p):
        plt.plot(
            _x,
            _y,
            linewidth=1.75,
            label='Test set ' + str(j)
        )
        j += 1
        _auc = auc(_x, _y)
        print(_auc)


    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.title('Precision Recall Curve', fontsize=17)
    plt.legend(loc='best')
    plt.show()
    plt.close()

    plt.figure(figsize=[14, 8])
    plt.title('Distribution of scores in Model 2', fontsize=17)
    plt.ylabel('Scores', fontsize=15)
    plt.xlabel('Samples', fontsize=15)
    _y = list(sorted(res))
    _x = list(range(len(_y)))
    plt.plot(
        _x,
        _y,
        linewidth=1.75
    )
    plt.show()
    plt.close()

# ---------------------------- #


if __name__ == "__main__":
    create_args()
    FLAGS.show_loss_fig = True
    tf.app.run(main)
