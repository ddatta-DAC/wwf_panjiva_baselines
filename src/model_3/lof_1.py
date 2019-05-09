import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
import pickle
import math
import tensorflow as tf
import numpy as np
import glob
import pandas as pd
import os
from sklearn.manifold import TSNE
import random
import inspect
import operator
import matplotlib.pyplot as plt
import sys
import time
from joblib import Parallel, delayed
from collections import OrderedDict

# try:
#     import model_3_v1
# except:
#     from . import model_3_v1 as model_3_v1


cur_path = '/'.join(
        os.path.abspath(
            inspect.stack()[0][1]
        ).split('/')[:-1]
    )

sys.path.append(cur_path)
# model_name = model_3_v1.MODEL_NAME
# SAVE_DIR = model_3_v1.SAVE_DIR
# OP_DIR = model_3_v1.OP_DIR
# DATA_DIR = model_3_v1.DATA_DIR
print(cur_path)
# print(OP_DIR)
KNN_K = 20
DISPLAY_ENSEMBLE_FIG = False


# ------------------------------------ #

def find_subspace_anomalies(x_id, x_emb, dim_count, show_figs=False) :
    global KNN_K
    time_1 = time.time()
    _count = np.random.randint(
        int(dim_count / 2),
        dim_count
    )

    print('count', _count)
    _dims = [int(_) for _ in (
        sorted(
            random.sample(
                list(range(0, dim_count)),
                _count)
        )
    )]

    sample_data = x_emb[:, _dims]
    print('Dimensions :', _dims)
    print(sample_data.shape)

    clf = LocalOutlierFactor(
        n_neighbors=KNN_K,
        contamination=0.01,
        metric='euclidean'
    )

    clf.fit(sample_data)
    X_scores = clf.negative_outlier_factor_

    z = np.array(X_scores) * -1
    if show_figs:
        y = sorted(
            z,
            reverse=True
        )
        x = list(range(len(y)))
        plt.plot(x, y)
        plt.title('Distribution of Local Outlier Factor Scores')
        plt.ylabel(' LOF score')
        plt.xlabel('Samples sorted by LOF score')
        plt.show()

        fig1, ax1 = plt.subplots()
        plt.title('Distribution of Negative Local Outlier Factor Scores')
        ax1.boxplot(X_scores)
        plt.show()

    print(X_scores.shape)

    X_scores = np.reshape(X_scores,-1)
    x_id = list(np.reshape(x_id,-1))


    _run_dict = {}

    for _id, _score in zip(x_id, X_scores):
        _run_dict[_id] = _score

    sorted_x = sorted(_run_dict.items(), key=operator.itemgetter(1))

    _id_list = [_[0] for _ in sorted_x]
    _scores = [_[1] for _ in sorted_x]
    result_dict = {
        k:v
        for k,v in zip(_id_list,_scores)
    }

    time_2 = time.time()
    print('time elapsed (Seconds):', time_2 - time_1)

    return result_dict


    #
    # print('Cut off', cut_off)
    # _candidates = []
    # for item in sorted_x:
    #     _candidates.append(item)


    # return _candidates
    # return result_dict

def anomaly_1( id_list, embed_list ):

    x_id = id_list
    x_emb = embed_list

    print(x_emb.shape)
    dim_count = int(x_emb.shape[-1])
    num_subspace_samples = int(dim_count*1.5)
    print('----')

    all_candidates = Parallel(
        n_jobs=num_subspace_samples
    )(
        delayed(find_subspace_anomalies)(
            x_id, x_emb, dim_count
        )
        for _ in range(num_subspace_samples)
    )

    # LOF scores are normalized , so simply add them up
    all_keys = list(x_id)
    print(len(all_keys))
    num_subsp = len(all_candidates)
    score_dict = {}

    for k in all_keys:
        s = 0
        for _r in all_candidates:
            s += _r[k]
        score_dict[k] = s/num_subsp

    if DISPLAY_ENSEMBLE_FIG:
        sorted_x = sorted(score_dict.items(), key=operator.itemgetter(1),reverse=True)
        y = [_[1] for _ in sorted_x ]
        x = list(range(len(y)))
        plt.figure(figsize=[12,12])
        plt.plot(x, y, 'r-')
        plt.title('Distribution of Ensemble Local Outlier Factor Scores')
        plt.ylabel(' LOF score')
        plt.xlabel('Samples sorted by LOF score')
        plt.show()

    # cut_off = np.percentile(list(score_dict.values()), 0.10)
    # print('Cut Off score :', cut_off)
    sorted_x = sorted(score_dict.items(), key=operator.itemgetter(1))
    print(len(sorted_x))

    id_list = [_[0] for _ in sorted_x]
    scores = [_[1] for _ in sorted_x]

    result_dict = {
        k: v
        for k, v in zip(id_list, scores)
    }

    # result_PId_list = []
    # for k,v in zip(id_list,scores):
    #     if v > cut_off:
    #         break
    #     result_PId_list.append(k)


    # print('Number of anomalies', len(result_PId_list))
    # with open(os.path.join(OP_DIR,'anomalies_1.pkl'),'wb') as fh:
    #     pickle.dump(result_PId_list,fh,pickle.HIGHEST_PROTOCOL)


    return result_dict




# anomaly_1()
# analysis_1()
