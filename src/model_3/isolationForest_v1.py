from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
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

DISPLAY_ENSEMBLE_FIG = False
# ------------------------ #

def anomaly_1( id_list, embed_list ):
    global DISPLAY_ENSEMBLE_FIG
    x_id = id_list
    x_emb = embed_list

    print(x_emb.shape)
    dim_count = int(x_emb.shape[-1])
    print('----')

    ensemble_if_obj = IsolationForest(
        n_estimators=300,
        max_samples=0.95,
        n_jobs=-1,
        random_state=None,
        verbose=1
    )

    ensemble_if_obj.fit(embed_list)
    scores = ensemble_if_obj.decision_function(embed_list)


    all_keys = list(x_id)
    score_dict = { k:v for k,v in zip(all_keys,scores)}
    sorted_x = sorted(score_dict.items(), key=operator.itemgetter(1))
    id_list = [_[0] for _ in sorted_x]
    scores = [_[1] for _ in sorted_x]

    result_dict = {
        k: v
        for k, v in zip(id_list, scores)
    }
    return result_dict

def find_subspace_anomalies(x_id, x_emb, dim_count, show_figs=False) :
    global KNN_K
    global USE_MAX

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

    ensemble_if_obj = IsolationForest(
        n_estimators=500,
        max_samples=0.95,
        n_jobs=-1,
        random_state=None,
        verbose=1
    )

    ensemble_if_obj.fit(x_emb)
    scores = ensemble_if_obj.decision_function(x_emb)

    all_keys = list(x_id)
    score_dict = {k: v for k, v in zip(all_keys, scores)}
    sorted_x = sorted(score_dict.items(), key=operator.itemgetter(1))

    _id_list = [_[0] for _ in sorted_x]
    _scores = [_[1] for _ in sorted_x]
    result_dict = {
        k: v
        for k, v in zip(_id_list, _scores)
    }

    return result_dict


def anomaly_2( id_list, embed_list ):
    global DISPLAY_ENSEMBLE_FIG
    x_id = id_list
    x_emb = embed_list

    print(x_emb.shape)
    dim_count = int(x_emb.shape[-1])
    print('----')

    num_subspace_samples = int(dim_count * 1)
    print('----')

    all_candidates = Parallel(
        n_jobs=num_subspace_samples
    )(
        delayed(find_subspace_anomalies)(
            x_id, x_emb, dim_count
        )
        for _ in range(num_subspace_samples)
    )

    all_keys = list(x_id)
    score_dict = {}
    for k in all_keys:
        s = []
        for _r in all_candidates:
            s.append(_r[k])
        score_dict[k] = np.min(s)


    sorted_x = sorted(score_dict.items(), key=operator.itemgetter(1))

    id_list = [_[0] for _ in sorted_x]
    scores = [_[1] for _ in sorted_x]

    result_dict = {
        k: v
        for k, v in zip(id_list, scores)
    }

    return result_dict