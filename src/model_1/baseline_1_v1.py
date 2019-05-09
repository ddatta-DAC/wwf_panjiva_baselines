import numpy as np
import yaml
import pandas as pd
import sklearn
from pprint import pprint
import glob
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import math
from sklearn import preprocessing
from scipy.stats import rv_discrete
import pickle
from sklearn.metrics import mutual_info_score
import itertools
import time
import sys

sys.path.append('./..')
sys.path.append('./../..')
from collections import OrderedDict
from joblib import parallel_backend
from joblib import Parallel, delayed

try:
    from .src.Eval import evaluation_v1
except:
    from src.Eval import evaluation_v1

try:
    import ad_tree_v1
except:
    from . import ad_tree_v1

import operator

# ------------------------- #
# Based on
# Detecting patterns of anomalies
# https://dl.acm.org/citation.cfm?id=1714140
# ------------------------- #
_author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "7.0"
# ------------------------- #

_DIR = None
DATA_DIR = None
CONFIG_FILE = 'config_1.yaml'
ID_LIST = None
SAVE_DIR = None
OP_DIR = None
config = None
DISCARD_0 = True

def get_data():
    global DATA_FILE
    global _DIR
    global DATA_DIR

    DATA_FILE = os.path.join(DATA_DIR, _DIR, 'train_x.pkl')
    with open(DATA_FILE, 'rb') as fh:
        DATA_X = pickle.load(fh)
    print(DATA_X.shape)
    _test_files = os.path.join(DATA_DIR, _DIR, 'test_x_*.pkl')
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


def setup(_dir=None):
    global CONFIG_FILE
    global _DIR
    global DATA_DIR
    global ID_LIST
    global DATA_X
    global OP_DIR
    global _DIR
    global SAVE_DIR
    global config

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    SAVE_DIR = config['SAVE_DIR']
    if _dir is None:
        _DIR = config['_DIR']
    else:
        _DIR = _dir
    OP_DIR = config['OP_DIR']
    DATA_DIR = config['DATA_DIR']
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    SAVE_DIR = os.path.join(SAVE_DIR, _DIR)

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)
    OP_DIR = os.path.join(OP_DIR, _DIR)

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    # DATA_FILE = os.path.join(DATA_DIR, _DIR, _DIR + '_x.pkl')
    # with open(DATA_FILE, 'rb') as fh:
    #     DATA_X = pickle.load(fh)
    # ID_LIST_FILE = os.path.join(DATA_DIR, _DIR, _DIR + '_idList.pkl')
    # with open(ID_LIST_FILE, 'rb') as fh:
    #     ID_LIST = pickle.load(fh)
    # print(DATA_X.shape)


# ----------------------------------- #
def calc_MI(x, y):
    # for now use this
    mi = mutual_info_score(x, y)
    return mi


# ----------------------------------- #

# Algorithm thresholds
MI_THRESHOLD = 0.1
ALPHA = 0.005


# Get arity of each domain
def get_domain_arity():
    f = os.path.join(DATA_DIR, _DIR, 'domain_dims.pkl')
    with open(f, 'rb') as fh:
        dd = pickle.load(fh)
    return list(dd.values())


# --------------- #
def get_MI_attrSetPair(data_x, s1, s2, obj_adtree):
    if len(s1) > 1 or len(s2) > 1: return 1

    if len(s1) == 1 or len(s2) == 1:
        _x = np.reshape(data_x[:, s1], -1)
        _y = np.reshape(data_x[:, s2], -1)
        return calc_MI(x=_x, y=_y)

    def _join(row, indices):
        r = '_'.join([str(row[i]) for i in indices])
        return r

    mask = np.random.choice([False, True], len(data_x), p=[0.8, 0.2])
    data_x = data_x[mask]

    _idx = list(s1)
    _idx.extend(s2)
    _atr = list(s1)
    _atr.extend(s2)
    _dict = {}
    for a in _atr:
        _dict[a] = set(data_x[:, [a]])

    _tmp_df = pd.DataFrame(data=DATA_X)
    _tmp_df = _tmp_df[_atr]

    _tmp_df['x'] = None
    _tmp_df['y'] = None
    _tmp_df['x'] = _tmp_df.apply(
        _join,
        axis=1,
        args=(s1,)
    )
    _tmp_df['y'] = _tmp_df.apply(
        _join,
        axis=1,
        args=(s2,)
    )
    mi = calc_MI(_tmp_df['x'], _tmp_df['y'])
    return mi


# MI = Sum ( P_(x)(y) log( P_(x)(y)/ P_(x)*P_(y) )


# get sets of attributes for computing r-value
# input attribute indices 0 ... m-1
# Returns sets of attributes of size k

def get_attribute_sets(
        attribute_list,
        obj_adtree,
        k=1
):
    global SAVE_DIR
    use_mi = True
    # check if file present in save dir
    op_file_name = 'set_pairs_' + str(k) + '.pkl'
    op_file_path = os.path.join(SAVE_DIR, op_file_name)

    if os.path.exists(op_file_path):
        with open(op_file_path, 'rb') as fh:
            set_pairs = pickle.load(fh)
        return set_pairs

    # -------------------------------------- #

    # We can attribute sets till size k
    # Add in size 1
    sets = list(itertools.combinations(attribute_list, 1))
    k = int(k)

    for _k in range(2, k + 1):
        _tmp = list(itertools.combinations(attribute_list, _k))
        sets.extend(_tmp)

    # check if 2 sets have MI > 0.1 and are mutually exclusive
    set_pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            s1 = sets[i]
            s2 = sets[j]
            print(s1, s2)
            # mutual exclusivity test
            m_e = (len(set(s1).intersection(s2)) == 0)
            mi_flag = False
            if m_e is False:
                continue
            # -- Ignore MI for now -- #
            # MI
            if use_mi is False:
                mi_flag = True
            else:
                mi = get_MI_attrSetPair(DATA_X, s1, s2, obj_adtree)
                if mi >= 0.1:
                    mi_flag = True

            if mi_flag is True:
                set_pairs.append((s1, s2))

    _dict = {e[0]: e[1] for e in enumerate(set_pairs, 0)}
    set_pairs = _dict
    # Save

    with open(op_file_path, 'wb') as fh:
        pickle.dump(set_pairs, fh, pickle.HIGHEST_PROTOCOL)

    return set_pairs


def get_count(obj_adtree, domains, vals):
    _dict = {k: v for k, v in zip(domains, vals)}
    res = obj_adtree.get_count(_dict)
    return res


def get_r_value(_id, record, obj_adtree, set_pairs, N):
    global ALPHA
    global DISCARD_0
    _r_dict = {}
    for k, v in set_pairs.items():
        _vals = []
        _domains = []
        for _d in v[0]:
            _domains.append(_d)
            _vals.append(record[_d])

        P_at = get_count(obj_adtree, _domains, _vals)+1
        P_at = P_at / (N+2)
        # print(P_at)

        _vals_1 = []
        _domains_1 = []
        for _d in v[1]:
            _domains_1.append(_d)
            _vals_1.append(record[_d])

        P_bt = get_count(obj_adtree, _domains_1, _vals_1)+1
        P_bt = P_bt / (N+2)
        # print(P_bt)

        _vals.extend(_vals_1)
        _domains.extend(_domains_1)

        P_ab = get_count(obj_adtree, _domains, _vals) / N
        r = (P_ab) / (P_at * P_bt)
        _r_dict[k] = r
    # heuristic
    sorted_r = list(sorted(_r_dict.items(), key=operator.itemgetter(1)))
    # print(sorted_r)

    score = 1
    U = set()
    threshold = ALPHA

    for i in range(len(sorted_r)):
        _r = sorted_r[i][1]
        tmp = set_pairs[sorted_r[i][0]]
        _attr = [item for sublist in tmp for item in sublist]

        if _r > threshold:
            break
        if  DISCARD_0 and _r <= 0.0 :
            continue

        if len(U.intersection(set(_attr))) == 0:
            U = U.union(set(_attr))
            score *= _r
    # print( _id, score)
    return _id, score


def main(_dir=None):
    global DATA_DIR
    global _DIR
    global config
    global OP_DIR
    global DATA_X
    global DISCARD_0

    _dir = _args['_dir']
    k_val = _args['k_val']

    DISCARD_0 =  _args['discard_0']

    setup(_dir)
    _DATA_X, test_anom_id, test_all_id, test_x = get_data()
    DATA_X = _DATA_X
    K = int(config['K'])
    # override
    if k_val is not None:
        K = k_val
    print(k_val)


    N = DATA_X.shape[0]
    obj_ADTree = ad_tree_v1.ADT()
    obj_ADTree.setup(DATA_X)

    attribute_list = list(range(DATA_X.shape[1]))
    print('Attribute list', attribute_list)

    attribute_set_pairs = get_attribute_sets(
        attribute_list,
        obj_ADTree,
        k=K
    )

    print(attribute_set_pairs)
    print(' Number of attribute set pairs ', len(attribute_set_pairs))

    # Testing phase

    number_CV = len(test_all_id)
    for n in range(number_CV):
        start = time.time()
        test_data = test_x[n]
        id_list = test_all_id[n]
        anom_id_list = test_anom_id[n]
        result_dict = {}


        results = []
        for _id, record in zip(id_list, test_data) :
            a = get_r_value(_id, record, obj_ADTree, attribute_set_pairs, N)
            results.append(a)

        for e in results:
            result_dict[e[0]] = e[1]

        end = time.time()
        print('-----------------------')
        print(_DIR)
        print('k = ', K)
        print(' Time taken :', end - start)
        # save file
        SAVE_FILE_OP = '_'.join([
            'result_alg_1_test_' + str(n),
            _DIR,
            str(time.time()).split('.')[0]
        ]) + '.pkl'

        SAVE_FILE_OP_PATH = os.path.join(DATA_DIR, SAVE_FILE_OP)
        with open(SAVE_FILE_OP_PATH, 'wb') as fh:
            pickle.dump(result_dict, fh, pickle.HIGHEST_PROTOCOL)

        tmp = sorted(result_dict.items(), key=operator.itemgetter(1))
        sorted_id_score_dict = OrderedDict()
        for e in tmp:
            sorted_id_score_dict[e[0]] = e[1]


        print('--------------------------')

        # Plot the distribution of r values
        _y = list(sorted(list(result_dict.values())))
        _x = list(range(len(_y)))

        plt.figure(figsize=[14, 8])
        plt.plot(
            _x,
            _y,
            color='red',
            linewidth=1.5
        )
        plt.xlabel('Samples (sorted)', fontsize=15)
        plt.ylabel('Decision value r', fontsize=15)

        f_name = 'r_vals' + '_K_'+ str(K) + '_test_' + str(n) + '_discard_0_'+ str(DISCARD_0) + '.png'
        f_path = os.path.join(OP_DIR, f_name)

        plt.savefig(f_path)
        plt.close()
        # -------------------------------#

        print('--------------------------')


        recall, precison = evaluation_v1.precision_recall_curve(
            sorted_id_score_dict,
            anomaly_id_list=anom_id_list
        )

        _auc = auc(recall, precison)
        plt.figure(figsize=[14, 8])
        plt.plot(
            recall,
            precison,
            color='blue', linewidth=1.75)
        plt.xlabel('Recall', fontsize=15)
        plt.ylabel('Precision', fontsize=15)
        plt.title('Recall | AUC ' + str(_auc), fontsize=15)
        f_name = 'precison-recall_1' + '_K_'+ str(K) + '_test_' + str(n) + '_discard_0_'+ str(DISCARD_0) + '.png'
        f_path = os.path.join(OP_DIR, f_name)
        plt.savefig(f_path)
        plt.close()

        print('----------------------------')


        x, y = evaluation_v1.performance_by_score(
            sorted_id_score_dict,
            anom_id_list
        )

        plt.figure(figsize=[14, 8])
        plt.plot(
            x,
            y,
            color='red', linewidth=1.75)
        # plt.xlabel(' ', fontsize=15)
        plt.ylabel('Percentage of anomalies detected', fontsize=15)
        plt.title('Lowest % of scores', fontsize=15)
        f_name = 'score_1_test_' + str(n) + '.png'
        f_path = os.path.join(OP_DIR, f_name)
        plt.savefig(f_path)
        plt.close()


# ------------------------------------------ #


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", nargs='?', default="None")
parser.add_argument("-k", "--k_val", nargs='?', default=1)
parser.add_argument("-r", "--discard_0", nargs='?', default=False)
args = parser.parse_args()

if args.dir == 'None':
    _dir = None
else:
    _dir = args.dir

if args.k_val == 1:
    k_val = None
else:
    k_val = args.k_val

_discard_0 = args.discard_0

_args = {
    '_dir' : _dir ,
    'k_val' : int(k_val),
    'discard_0' : bool(_discard_0)
}

main(_args)