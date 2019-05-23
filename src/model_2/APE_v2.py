# this is an improvement over APE_V1
# key point : pure unsupervised

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

try:
    from src.model_2 import APE_tf_model_1
except:
    from .src.model_2 import APE_tf_model_1


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
        6,
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
        'model_ape_10_k_3_frozen.pb',
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
    print(dd)
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
APE_tf_model_1._DIR = _DIR

APE_tf_model_1.OP_DIR = OP_DIR
if not os.path.exists(OP_DIR):
    os.mkdir(OP_DIR)
MODEL_NAME = 'model_ape'

embedding_dims = [16]
domain_dims = get_domain_arity()
cur_path = get_cur_path()



# ----------------------------------------- #

# ---------------------------- #
# Return data for training
# x_pos_inp : [?, num_entities]
# x_neg_inp = [?,neg_samples, num_entities]

def get_training_data(
        data_x,
        neg_samples,
        index
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

    TRAIN_DATA_FILE_NAME = 'ape_v1_train_data_'+ str(neg_samples) + str(index) + '.pkl'
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
    count_test_sets = len(test_x)






    test_result_r = []
    test_result_p = []
    res = None
    for i in range(count_test_sets):

        train_data_x = np.vstack([data_x, test_x[i]])
        data, inp_dims = get_training_data(
            train_data_x,
            FLAGS.neg_samples,
            index=i
        )

        num_domains = len(inp_dims)
        model_obj = APE_tf_model_1.model_ape_1(MODEL_NAME)
        model_obj.set_model_params(
            num_entities=num_domains,
            inp_dims=inp_dims,

            neg_samples=FLAGS.neg_samples,
            batch_size=FLAGS.batchsize,
            num_epochs=FLAGS.num_epochs,
            lr= FLAGS.learning_rate,
            chkpt_dir=checkpoint_dir
        )

        model_obj.set_hyper_parameters(
            emb_dims=[10],
            use_bias=[True, False]
        )

        print(FLAGS.use_pretrained)
        if FLAGS.use_pretrained is False:
            model_obj.build_model()
            model_obj.train_model(data)

        _x = test_x[i]
        _x = np.vstack([_x,data_x] )
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
    mean_auc = 0
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
        mean_auc += _auc

    mean_auc = mean_auc/len(test_result_r)
    print('Mean ', mean_auc)
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
