import numpy as np
import pandas
import sklearn
import operator
import os
import pickle
import glob
import argparse


DATA_DIR = '../../Data'
_DIR = 'data_2'

DATA_FILE = None


# ------------------------------------------------- #
# return a list of all the data  along with list of test files(for matching)
# ------------------------------------------------- #

def get_data_1():
    global DATA_FILE
    global _DIR
    global DATA_DIR

    DATA_FILE = os.path.join(DATA_DIR, 'train_x.pkl')
    with open(DATA_FILE, 'rb') as fh:
        x = pickle.load(fh)

    _test_files = os.path.join(
        DATA_DIR,
        'test_x_*.pkl'
    )

    test_files = glob.glob(_test_files)
    print(test_files)
    all_data = [x]
    for t in test_files:
        with open(t, 'rb') as fh:
            data = pickle.load(fh)
            _test_x = data[2]
            all_data.append(_test_x)

    return all_data, test_files


def get_data_2(test_file_list):
    global DATA_FILE
    global _DIR
    global DATA_DIR

    DATA_FILE = os.path.join(DATA_DIR, 'train_x.pkl')

    with open(DATA_FILE, 'rb') as fh:
        train_x = pickle.load(fh)

    # _test_files = os.path.join(
    #     DATA_DIR,
    #     'test_x_*.pkl'
    # )
    # print(_test_files)
    # test_files = glob.glob(_test_files)

    test_x = []
    for t in test_file_list:
        with open(t, 'rb') as fh:
            data = pickle.load(fh)
            test_x.append(data[2])
    return train_x, test_x


def get_domain_dims(data_dir):
    f_path = os.path.join(data_dir, 'domain_dims.pkl')
    with open(f_path, 'rb') as fh:
        res = pickle.load(fh)
    print(res)
    return list(res.values())


# --------------------------------------------------- #
# Data here is a combination a complete set of train + test
# --------------------------------------------------- #
def get_probabilities(data_dir, data):
    res_dict_arr = []
    _domain_dims = get_domain_dims(data_dir)

    for d in range(len(_domain_dims)):
        print('-----')
        col_x = data[:, d]
        print('Domain : ', d)
        elements = list(range(_domain_dims[d]))
        _res = {}
        _sum = 0

        for e in elements:
            c = list(col_x).count(e)
            if c == 0:
                c = 1
                print('0 count found')
            _res[e] = c
            _sum += c
        _res = {k: v / _sum for k, v in _res.items()}
        print(_res)
        res_dict_arr.append(_res)
        print('-----')

    return res_dict_arr


# ---- #
def get_record_entity_probability(arr, _dict):
    arr = [_dict[_a] for _a in arr]
    return arr


def setup_entity_prob_data(
        numeric_data,
        ep_f_name,
        prob_dict
):
    global DATA_DIR
    domains_dims = get_domain_dims(DATA_DIR)

    prob_x = np.zeros(np.shape(numeric_data))
    print(prob_x.shape)

    for d in range(len(domains_dims)):
        _prob_dict_domain = prob_dict[d]
        _y = numeric_data[:, d]

        r = get_record_entity_probability(_y, _prob_dict_domain)
        r = np.array(r)
        prob_x[:, d] = r

    # save file
    f_path = os.path.join(DATA_DIR, ep_f_name)
    with open(f_path, 'wb') as fh:
        pickle.dump(prob_x, fh, pickle.HIGHEST_PROTOCOL)

    return


def main(argv):

    global DATA_DIR
    global _DIR

    _DIR = argv['_dir']
    DATA_DIR = os.path.join(DATA_DIR,_DIR)


    # data_x is a list [ train, test_!, test_2,... ]
    data_x , test_file_list = get_data_1()

    train_x, test_x, = get_data_2(test_file_list)

    for i  in range(len(test_file_list)):
        cur_test_x = data_x[i+1]
        cur_train_x = data_x[0]
        combined = np.vstack([cur_train_x,cur_test_x])

        # probability dictionary for this data-set
        prob_dict = get_probabilities(DATA_DIR, combined)

        # numeric_data = [train_x]
        # numeric_data.extend(test_x)

        t_f = test_file_list[i]
        _serialID = (t_f.split('.')[-2]).split('_')[-1]

        numeric_data = np.vstack([cur_train_x,cur_test_x])
        ep_f_name = 'entity_prob_test_x_' + _serialID +'.pkl'
        setup_entity_prob_data(
            numeric_data,
            ep_f_name,
            prob_dict
        )

        # file_names = ['entity_prob_train_x.pkl']

    # for t_f in test_files:
    #
    #     print('>>', t_f, k)
    #     fn =
    #     file_names.append(fn)

    return


# ----------------------------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", nargs='?', default="None")
args = parser.parse_args()

if args.dir == 'None':
    _dir = 'data_3'
else:
    _dir = args.dir

_args = {
    '_dir': _dir
}
main(_args)

