# --------------------------------- #
# Calculate  record embeddings based on Arora Paper
# --------------------------------- #
# this is to verify for both sets
# ----------------------------------- #

import operator
import pickle
import numpy as np
import glob
import json
import pandas as pd
import os
import argparse
import sys
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import time
import inspect
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
import yaml
from sklearn.metrics import auc

sys.path.append('./..')
sys.path.append('./../../.')
try:
    from .src.Eval import evaluation_v1
except:
    from src.Eval import evaluation_v1

try:
    from .src.Eval import evaluation_v2
except:
    from src.Eval import evaluation_v2

try:
    from .src.model_3 import lof_1
except:
    from src.model_3 import lof_1

try:
    from .src.model_3 import isolationForest_v1 as IF
except:
    from src.model_3 import isolationForest_v1 as IF

try:
    from .src.model_5_verify import my_model_v_1_3 as tf_model
except:
    from src.model_5_verify import my_model_v_1_3 as tf_model

# ------------------------------------ #
cur_path = '/'.join(
    os.path.abspath(
        inspect.stack()[0][1]
    ).split('/')[:-1]
)

sys.path.append(cur_path)

_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "5.0"
__processor__ = 'embedding'
_SAVE_DIR = 'save_dir'
MODEL_NAME = 'model_5_v1'
_DIR = None
DATA_DIR = None
MODEL_OP_FILE_PATH = None
CONFIG_FILE = 'model_5_config.yaml'
CONFIG = None


# ----------------------------------------- #

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

# embedding_dims = None
DOMAIN_DIMS = None


def setup_general_config():
    global MODEL_NAME
    global _DIR
    global SAVE_DIR
    global OP_DIR
    global _SAVE_DIR
    global CONFIG

    SAVE_DIR = os.path.join(CONFIG['SAVE_DIR'], _DIR)
    OP_DIR = os.path.join(CONFIG['OP_DIR'], _DIR)
    print(cur_path)
    print(OP_DIR)

    if not os.path.exists(CONFIG['OP_DIR']):
        os.mkdir(CONFIG['OP_DIR'])

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    if not os.path.exists(CONFIG['SAVE_DIR']):
        os.mkdir(os.path.join(CONFIG['SAVE_DIR']))

    if not os.path.exists(SAVE_DIR):
        os.mkdir(os.path.join(SAVE_DIR))
    return


# --------------------------------------------- #

def set_up_model(config, _dir):
    global embedding_dims
    global SAVE_DIR
    global OP_DIR
    global MODEL_NAME

    if type(config[_dir]['op_dims']) == str:
        embedding_dims = config[_dir]['op_dims']
        embedding_dims = embedding_dims.split(',')
        embedding_dims = [int(e) for e in embedding_dims]
    else:
        embedding_dims = config[_dir]['op_dims']
    # print(embedding_dims)

    model_obj = tf_model.model(MODEL_NAME, SAVE_DIR, OP_DIR)
    model_obj.set_model_options(
        show_loss_figure=config[_dir]['show_loss_figure'],
        save_loss_figure=config[_dir]['save_loss_figure'],
        set_w_mean=True
    )

    domain_dims = get_domain_dims()
    LR = config[_dir]['learning_rate']
    model_obj.set_model_hyperparams(
        domain_dims=domain_dims,
        emb_dims=embedding_dims,
        batch_size=config[_dir]['batchsize'],
        num_epochs=config[_dir]['num_epochs'],
        learning_rate=LR,
        alpha=config[_dir]['alpha']
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

    test_files = glob.glob(_test_files)
    print(test_files)
    test_x = []
    test_anom_id = []
    test_all_id = []
    for t in test_files:
        print(t)
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
    entity_prob_train_x = None
    # with open(entity_prob_train_file, 'rb') as fh:
    #     entity_prob_train_x = pickle.load(fh)
    #     print(entity_prob_train_x.shape)
    #     print('---')

    entity_prob_test = []
    test_SerialID = []

    for t in test_files:
        i = ((t.split('/')[-1]).split('.')[-2]).split('_')[-1]
        print(t, i)
        _entity_prob_test_file = os.path.join(
            DATA_DIR,
            'entity_prob_test_x_' + str(int(i)) + '.pkl'
        )
        test_SerialID.append(i)
        # entity_prob_test_file = glob.glob(_entity_prob_test_file)
        with open(_entity_prob_test_file, 'rb') as fh:
            data = pickle.load(fh)
            entity_prob_test.append(data)
    print([_.shape for _ in entity_prob_test])

    return DATA_X, test_anom_id, test_all_id, test_x, train_ids, entity_prob_train_x, entity_prob_test, test_SerialID


def domain_apha_aux():
    global _DIR
    global DATA_DIR

    _data_file = os.path.join(DATA_DIR, 'train_x.pkl')
    with open(_data_file, 'rb') as fh:
        x = pickle.load(fh)

    _test_files = os.path.join(
        DATA_DIR,
        'test_x_*.pkl'
    )
    print(_test_files)
    test_files = glob.glob(_test_files)

    for t in test_files:
        with open(t, 'rb') as fh:
            data = pickle.load(fh)
            _test_x = data[2]
            x = np.vstack([x, _test_x])
    print(x.shape)
    return x


def get_domain_alpha():
    global _DIR
    global DATA_DIR

    _data = domain_apha_aux()
    domain_dims = get_domain_dims()
    num_domains = len(domain_dims)
    domain_alpha = []
    for i in range(num_domains):
        tmp = _data[:, i]
        df = pd.Series(tmp).value_counts()
        df.sort_index(inplace=True)
        vals = list(df.sort_values().values)
        _alpha = (np.max(vals) * np.min(vals)) / (1 + np.min(vals))
        _alpha = np.median(vals)
        domain_alpha.append(_alpha)
    return domain_alpha


def process(
        idx,
        CONFIG,
        _DIR,
        data_x,
        test_x,
        train_ids,
        test_all_id,
        test_anom_id,
        test_SerialID,
        entity_prob_test,
        eval_type
):
    model_obj = set_up_model(CONFIG, _DIR)
    _x = np.vstack([data_x, test_x[idx]])
    model_obj.set_SerialID(test_SerialID[idx])
    _use_pretrained = CONFIG[_DIR]['use_pretrained']

    if _use_pretrained is True:
        saved_file_path = None

        pretrained_file = CONFIG[_DIR]['saved_model_file']
        if type(pretrained_file) == list:
            _match = '_serialID_' + str(test_SerialID[idx])
        _pretrained_file = None
        _match = '_serialID_' + str(test_SerialID[idx])

        if type(pretrained_file) == list:
            # search for the one that matches test_SerialID
            for _p in pretrained_file:
                if _match in _p:
                    _pretrained_file = _p
                    break

            print('Pretrained File :', _pretrained_file)
            saved_file_path = os.path.join(
                SAVE_DIR,
                'checkpoints',
                _pretrained_file
            )
        elif pretrained_file is False:
            # Find the pretrained file
            __fname = '*' + model_obj.model_signature + '*' + _match + '*.pb'
            try:
                saved_file_path = glob.glob(os.path.join(
                    SAVE_DIR,
                    'checkpoints',
                    __fname
                ))[0]
            except:
                saved_file_path = None

        if saved_file_path is not None:
            model_obj.set_pretrained_model_file(saved_file_path)
        else:
            model_obj.train_model(data_x)

    elif _use_pretrained is False:
        model_obj.train_model(data_x)

    _ep = entity_prob_test[idx]
    if CONFIG[_DIR]['w_mean']:
        mean_embeddings = model_obj.get_w_embedding_mean(_x, _ep)
    else:
        mean_embeddings = model_obj.get_embedding_mean(_x)

    _test_all_id = test_all_id[idx]
    _all_ids = list(train_ids)
    _all_ids.extend(list(_test_all_id))

    anomalies = test_anom_id[idx]

    # USE LOF here
    sorted_id_score_dict = lof_1.anomaly_1(
        id_list=_all_ids,
        embed_list=mean_embeddings
    )

    _scored_dict_test = {}

    for k1, v in sorted_id_score_dict.items():
        if k1 in _test_all_id or k1 in _scored_dict_test:
            _scored_dict_test[k1] = v

    if eval_type == 1:
        recall, precison = evaluation_v1.precision_recall_curve(
            _scored_dict_test,
            anomaly_id_list=anomalies
        )
    elif eval_type == 2:
        recall, precison = evaluation_v2.precision_recall_curve(
            _scored_dict_test,
            anomaly_id_list=anomalies
        )

    # test_result_r.append(recall)
    # test_result_p.append(precison)
    cur_auc = auc(recall, precison)
    print('AUC ::', cur_auc)
    print('--------------------------')
    return cur_auc, recall, precison


def plot_all_PR(test_result_r, test_result_p, OP_DIR, eval_type=1):
    plt.figure(figsize=[14, 8])
    j = 1
    res_str = 'auPR : '
    auc_avg = 0
    for _x, _y in zip(test_result_r, test_result_p):
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
        auc_avg += _auc

    auc_avg = auc_avg / len(test_result_r)
    res_str += '| Avg ' + "{0:.2f}".format(auc_avg)
    print('Average ', auc_avg)

    plt.xlabel('Recall', fontsize=15)
    plt.ylabel('Precision', fontsize=15)
    plt.title('Precision Recall Curve ' + res_str, fontsize=18)
    plt.legend(loc='best')

    f_name = None
    if eval_type == 1:
        f_name = 'precison-recall_test_' + str(time.time()).split('.')[0] + '.png'
    elif eval_type == 2:
        f_name = 'precison-recall_test_' + str(time.time()).split('.')[0] + '_type_2' + '.png'
    f_path = os.path.join(OP_DIR, f_name)
    plt.savefig(f_path)
    # plt.show()
    plt.close()
    return

    # ------------------
    # Log
    # ------------------


def log_results(CONFIG, _DIR, OP_DIR, auc):
    log_file = CONFIG[_DIR]['log_file']
    log_file_path = os.path.join(OP_DIR, log_file)
    df = None
    print(log_file_path)
    if os.path.exists(log_file_path):
        df = pd.read_csv(log_file_path, index_col=None)
        print(df)

    _dict = {
        'auc': auc
    }
    for k, v in CONFIG[_DIR].items():
        if k in ['log_file', 'saved_model_file']:
            v = None
        if v is not None and type(v) == list:
            v = ';'.join([str(_) for _ in v])

        _dict[k] = str(v)

    if df is not None:

        df = df.append(_dict, ignore_index=True)
    else:
        _dict = {k: [v] for k, v in _dict.items()}
        df = pd.DataFrame(_dict)
    print(df)
    df.to_csv(log_file_path, index=False)

    return


def main():
    global embedding_dims
    global SAVE_DIR
    global _DIR
    global DATA_DIR
    global CONFIG
    global CONFIG_FILE
    global MODEL_NAME
    global DOMAIN_DIMS

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    _DIR = CONFIG['_DIR']
    DATA_DIR = CONFIG['DATA_DIR'] + '/' + _DIR
    setup_general_config()

    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(os.path.join(SAVE_DIR, 'checkpoints'))

    # ------------ #

    data_x, test_anom_id, test_all_id, test_x, train_ids, entity_prob_train_x, entity_prob_test, test_SerialID = get_data()

    DOMAIN_DIMS = get_domain_dims()
    print(data_x.shape)
    print([_.shape for _ in test_x])
    print([_.shape for _ in entity_prob_test])

    eval_type = CONFIG['eval_type']

    lof_1.KNN_K = CONFIG[_DIR]['lof_K']
    from joblib import Parallel, delayed

    # ------------
    # 10 test cases
    # ------------

    all_res = Parallel(n_jobs=CONFIG['num_jobs'])(
        delayed(process)(
            idx,
            CONFIG,
            _DIR,
            data_x,
            test_x,
            train_ids,
            test_all_id,
            test_anom_id,
            test_SerialID,
            entity_prob_test,
            eval_type)

        for idx in range(len(test_x))
    )

    all_auc = [_[0] for _ in all_res]
    test_result_r = [_[1] for _ in all_res]
    test_result_p = [_[2] for _ in all_res]

    print('Mean AUC', np.mean(all_auc))
    mean_auc = np.mean(all_auc)
    # log_results(CONFIG, _DIR, OP_DIR, auc)


# ----------------------------------------------------------------- #
# find out which model works best
# ----------------------------------------------------------------- #
def run_experiment(
        _dir=None ,
        exp_dict = None
):
    global embedding_dims
    global SAVE_DIR
    global _DIR
    global DATA_DIR
    global CONFIG
    global CONFIG_FILE
    global MODEL_NAME
    global DOMAIN_DIMS

    with open(CONFIG_FILE) as f:
        CONFIG = yaml.safe_load(f)

    if _dir is not None:
        CONFIG['_DIR'] = _dir
        _DIR = _dir

    _DIR = CONFIG['_DIR']
    DATA_DIR = CONFIG['DATA_DIR'] + '/' + _DIR
    setup_general_config()

    if not os.path.exists(os.path.join(SAVE_DIR, 'checkpoints')):
        os.mkdir(os.path.join(SAVE_DIR, 'checkpoints'))

    # ------------ #

    data_x, test_anom_id, test_all_id, test_x, train_ids, entity_prob_train_x, entity_prob_test, test_SerialID = get_data()
    DOMAIN_DIMS = get_domain_dims()
    print(data_x.shape)
    print([_.shape for _ in test_x])
    print([_.shape for _ in entity_prob_test])

    eval_type = CONFIG['eval_type']

    from joblib import Parallel, delayed

    # ------------
    # 10 test cases
    # ------------
    max_count = CONFIG['num_jobs']

    for exp_emb_size in exp_dict[_DIR]['exp_emb_size']:
        CONFIG[_DIR]['op_dims'] = [8, exp_emb_size]

        for alpha in exp_dict[_DIR]['alpha']:
            CONFIG[_DIR]['alpha'] = alpha

            for lof_k in exp_dict[_DIR]['lof_k']:
                CONFIG[_DIR]['lof_K'] = lof_k
                lof_1.KNN_K = CONFIG[_DIR]['lof_K']
                all_res = Parallel(n_jobs=CONFIG['num_jobs'])(
                    delayed(process)(
                        idx,
                        CONFIG,
                        _DIR,
                        data_x,
                        test_x,
                        train_ids,
                        test_all_id,
                        test_anom_id,
                        test_SerialID,
                        entity_prob_test,
                        eval_type)

                    for idx in range(max_count)
                )
                all_auc = [_[0] for _ in all_res]

                print('Mean AUC', np.mean(all_auc))
                mean_auc = np.mean(all_auc)
                log_results(CONFIG, _DIR, OP_DIR, mean_auc)


# ----------------------------------------------------------------- #

# main()

exp_dict = {
    'data_1': {
        'exp_emb_size': [6, 8, 10, 12, 14, 16],
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001],
        'lof_k': [10, 15, 18, 20, 22, 24, 30]

    },
    'data_2': {
        'exp_emb_size': [6, 8, 10, 12, 14, 16],
        'lof_k': [10, 15, 18, 20, 22, 24, 30],
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001]
    },
    'data_5': {
        'exp_emb_size': [4, 6, 8, 10],
        'lof_k': [10, 12, 15, 18, 20, 22, 24],
        'alpha': [1, 0.1, 0.01, 0.001, 0.0001]
    }

}



parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", nargs='?', default="None")
args = parser.parse_args()

if args.dir == 'None':
    _dir = 'data_3'
else:
    _dir = args.dir
#
# run_experiment(_dir=_dir , exp_dict = exp_dict)

main()