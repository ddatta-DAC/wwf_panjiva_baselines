import matplotlib.pyplot as plt
import time
import os
import sys
import time
import seaborn as sns
import glob
from collections import OrderedDict
import numpy as np
import pandas as pd
import logging
import yaml
import pickle
import inspect
import sys
import operator
from sklearn.metrics import auc

sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../.')

cur_path = '/'.join(
    os.path.abspath(
        inspect.stack()[0][1]
    ).split('/')[:-2]
)
sys.path.append(cur_path)

try:
    from .src.Eval import evaluation_v1 as eval
except:
    from src.Eval import evaluation_v1 as eval

try:
    from src.CompreX.comprex.comprex import CompreX
except:
    from .src.CompreX.comprex.comprex import CompreX

CONFIG_FILE = 'config_compreX.yaml'
with open(CONFIG_FILE) as f:
    config = yaml.safe_load(f)

MODEL_NAME = None
OP_DIR = None
DATA_DIR = None
SAVE_DIR = None
_DIR = None


# --------------------------- #

def trivial_test():
    X = pd.DataFrame([
        ['a', 'b', 'x'],
        ['a', 'b', 'x'],
        ['a', 'b', 'x'],
        ['a', 'b', 'x'],
        ['a', 'c', 'x'],
        ['a', 'c', 'y'],
        ['a', 'b', 'x']
    ],
        columns=['f1', 'f2', 'f3'],
        index=[i for i in np.arange(7, 14)],
        dtype='category'
    )

    estimator = CompreX(logging_level=logging.ERROR)
    estimator.transform(X)
    estimator.fit(X)
    res = estimator.predict(X)
    print(type(res))

    print(res)


# --------------------------- #

def setup():
    global SAVE_DIR
    global _DIR
    global config
    global OP_DIR
    global MODEL_NAME
    global DATA_DIR
    global domain_dims
    global cur_path

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
    MODEL_NAME = 'compreX'


# --------
# Ensure the columns have different values.
# Append column id to each value
# --------
def get_data():
    global _TIME_IT
    global DATA_X_FILE
    global _DIR
    global DATA_DIR

    def stringify_data(arr) -> np.array:
        tmp1 = []
        for i in range(arr.shape[0]):
            tmp2 = []
            for j in range(arr.shape[1]):
                tmp2.append(str(arr[i][j]) + '_' + str(j))
            tmp1.append(tmp2)

        tmp1 = np.array(tmp1)
        return tmp1

    DATA_X_FILE = os.path.join(DATA_DIR, _DIR, 'train_x.pkl')
    with open(DATA_X_FILE, 'rb') as fh:
        DATA_X = pickle.load(fh)

    DATA_X = stringify_data(DATA_X)
    print(DATA_X.shape)


    DATA_Xid_FILE = os.path.join(DATA_DIR, _DIR, 'train_x_id.pkl')
    with open(DATA_Xid_FILE, 'rb') as fh:
        DATA_X_id = pickle.load(fh)

    print(len(DATA_X_id))

    _test_files = os.path.join(
        DATA_DIR,
        _DIR,
        'test_x_*.pkl'
    )
    test_files = sorted(glob.glob(_test_files))
    print(test_files)

    # test data is the data (N*M), needs to be appended to DATA_X
    test_x = []
    test_anom_id = []
    test_all_id = []

    for t in test_files:
        print(t)
        with open(t, 'rb') as fh:
            data = pickle.load(fh)
            test_anom_id.append(data[0])
            test_all_id.append(data[1])
            _tmp = data[2]
            _tmp = stringify_data(_tmp)
            test_x.append(_tmp)
            print(data[0].shape, len(data[1]), _tmp.shape)

    return DATA_X, DATA_X_id, test_anom_id, test_all_id, test_x


# --------------------------- #
def main():
    global _TIME_IT
    global _DIR
    global OP_DIR
    global SAVE_DIR
    global config
    setup()


    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
        if os.path.exists(os.path.join(SAVE_DIR, _DIR)):
            os.mkdir(os.path.join(SAVE_DIR, _DIR))

    checkpoint_dir = os.path.join(SAVE_DIR)
    print(os.getcwd())

    # data_x, test_anom_id, test_all_id, test_x =
    data_x, data_x_id, test_anom_id, test_all_id, test_x = get_data()
    count_test_sets = min(len(test_x),1)

    test_result_r = []
    test_result_p = []
    res = None
    time_arr = []
    auc_arr = []
    for i in range(count_test_sets):
        start_time = time.time()

        _x = test_x[i]
        _x = np.vstack([data_x, _x])

        test_ids = test_all_id[i]
        print(' >> ', len(test_ids))
        _x_id = list(data_x_id)
        _x_id.extend(test_ids)

        print(_x.shape)
        # _x = _x[:2000, :4]
        # _x_id = _x_id[:2000]

        print(_x.shape)
        print(len(_x_id))
        print(_x)

        # known anomalies
        anomaly_ids = test_anom_id[i]

        # ---- Core ------ #
        _df_input = []
        for _j in range(_x.shape[0]):
            _df_input.append(list(_x[_j]))

        cols = ['f' + str(j) for j in range(_x.shape[1])]
        X = pd.DataFrame(
            _df_input,
            columns=cols,
            index=[_j for _j in range(_x.shape[0])],
            dtype='category'
        )

        estimator = CompreX(logging_level=logging.ERROR)
        estimator.transform(X)
        estimator.fit(X)
        res = estimator.predict(X)
        '''
            'res' is ordered in the order of the input
            match it with the ordered list of ids
        '''
        anomaly_scores = list(res)
        anomaly_score_dict = { k:v for k,v in zip(_x_id,anomaly_scores) }

        # --------------- #
        ''' 
        Sort in reverse order, since higher score means anomaly 
        '''

        tmp = sorted(
            anomaly_score_dict.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        sorted_id_score_dict = OrderedDict()
        for e in tmp:
            sorted_id_score_dict[e[0]] = e[1]

        recall, precison = eval.precision_recall_curve(
            sorted_id_score_dict,
            anomaly_id_list=anomaly_ids
        )
        end_time = time.time()
        time_taken = end_time - start_time
        _auc = auc(recall, precison)

        print('Test case ', i , 'Time taken [seconds]', time_taken , 'AUC',  _auc)
        print('--------------------------')
        time_arr.append(time_taken)
        auc_arr.append(_auc)



    print('=================')
    print('Avg AUC :', np.mean(auc_arr))
    print('Avg time', np.mean(time_taken))


    '''
    if _TIME_IT == False:
    _auc = auc(recall, precison)
    print('AUC', _auc)
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

    
    '''
    print('----------------------------')
    '''

    plt.figure(figsize=[14, 8])
    j = 1
    mean_auc = 0
    all_auc = []
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
        all_auc.append(_auc)
    mean_auc = np.mean(all_auc)

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

    # plt.show()
    plt.close()
    if _TIME_IT == False:
        # save the results
        _dict = {
            'mean_auc': mean_auc,
            'all_auc': ';'.join([str(_) for _ in all_auc])
        }
        for k, v in config[_DIR]:
            _dict[k] = str(v)

        _dict = {k: [v] for k, v in _dict.items()}
        df = pd.DataFrame(_dict)

        res_fname = 'ape_result' + str(time.time()).split('.')[0] + '.csv'
        df.to_csv(
            os.path.join(OP_DIR, res_fname)
        )
    if _TIME_IT:
        print('Time Taken :', end_time - start_time)
        
    
    '''

# ---------------------------- #


# if __name__ == "__main__":
#     create_args()
#     FLAGS.show_loss_fig = True
#     tf.app.run(main)


main()

