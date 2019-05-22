import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
import matplotlib.pyplot as plt


def precision_recall_curve(
        sorted_id_score_dict,
        anomaly_id_list
):
    recall = 0
    correct = 0
    recall_vals = []
    precision_vals = []
    num_anomalies = len(anomaly_id_list)
    print('Number of Anomalies ', num_anomalies)
    total_count = len(sorted_id_score_dict)
    print(total_count)
    input_ids = list(sorted_id_score_dict.keys())
    _c1 =  list(sorted_id_score_dict.keys())
    _c2 =  list(sorted_id_score_dict.values())
    _tmp = np.transpose(np.array([_c1,_c2]))
    df = pd.DataFrame(data=_tmp)
    print(df)

    import math
    def log_xform(row):
        a = row[1]
        if a < 0:
            return -math.log(-a,10)
        else :
            return 0

    # df[1] = df.apply(log_xform,axis=1)
    # df[1]
    print(df)
    steps = 10000
    t = (np.max(df[1]) - np.min(df[1]))/steps

    start =  np.min(df[1])

    # Log the scores
    # Set thresholds t
    # points 0.5 to 100

    # Following charu aggarwal
    #   Precision  = (S(t) intersection G) /  |S(t)|
    #   Recall  = (S(t) intersection G) /  |G|

    print('------')
    cur_score_val = start
    cur_idx = 0

    _ids = df[0]
    _scores = df[1]


    while(cur_score_val <= np.max(_scores)):
        j = cur_idx
        while _scores[j] < cur_score_val:
            j += 1
        # print('Cur idx ',cur_idx)
        cur_idx = j
        res_set = set(_ids[:cur_idx+1])
        _numerator = len(set(res_set).intersection(anomaly_id_list))
        p = _numerator / (cur_idx+1)
        r = _numerator / num_anomalies
        precision_vals.append(p)
        recall_vals.append(r)
        cur_score_val += t
        if r == 1.0 :
            break

    # print(recall_vals)
    # print(precision_vals)

    # for t in np.arange(0.25, 100 + 0.125, 0.125):
    #     _k = int((t/100)*total_count)
    #     _numerator = len(set(input_ids[:_k]).intersection(anomaly_id_list))
    #     # print(_numerator, _k ,num_anomalies)
    #     p = _numerator/_k
    #     r = _numerator/num_anomalies
    #     precision_vals.append(p)
    #     recall_vals.append(r)

        # -------------------------- #

    return recall_vals, precision_vals


sorted_id_score_dict = {
    12:-2125,
    23:-2000,
    25:-1255,
    24:-987,
    918:-879,
    147:-877,
    54:-10.5,
    10:-1.8,
    98:-1.0,
    74:-1.0,
    200:-1.0
}

anomaly_id_list = [12,25,23,24,54,98,147]
# precision_recall_curve(
#         sorted_id_score_dict,
#         anomaly_id_list)