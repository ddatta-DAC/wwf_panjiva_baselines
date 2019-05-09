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

    # Set thresholds t
    # points 0.5 to 100

    # Following charu aggarwal
    #   Precision  = (S(t) intersection G) /  |S(t)|
    #   Recall  = (S(t) intersection G) /  |G|

    print('------')
    for t in np.arange(0.25, 100 + 0.125, 0.125):
        _k = int((t/100)*total_count)
        _numerator = len(set(input_ids[:_k]).intersection(anomaly_id_list))
        print(_numerator, _k ,num_anomalies)
        p = _numerator/_k
        r = _numerator/num_anomalies
        precision_vals.append(p)
        recall_vals.append(r)

        # -------------------------- #

    # cur_count = 0
    # Assumption is that the lowest likelihood events are anomalous
    # for id, score in sorted_id_score_dict.items():
    #     flag = False
    #     if id in anomaly_id_list:
    #         flag = True
    #     cur_count += 1
    #
    #     if flag is True:
    #         recall+=1
    #         correct += 1
    #         print('correct / num_anomalies / cur_count',
    #               correct,
    #               num_anomalies,
    #               cur_count
    #               )
    #     p = correct/cur_count
    #     r = recall/num_anomalies
    #
    #     precision_vals.append(p)
    #     recall_vals.append(r)
    #     if r == 1.0 :
    #         break

    # x Axis : Recall
    # Y Axis : Precision

    return recall_vals, precision_vals

def performance_by_score(
        sorted_id_score_dict,
        anomaly_id_list
    ):

    N = len(sorted_id_score_dict)
    # increase by 0.5 % from 0.5 to 5
    _dict = {}
    for n in np.arange(0.5,5.5,0.5):
        c = int(N*n)
        _list_1 = list(sorted_id_score_dict.keys())[:c]
        res = len(set(_list_1).intersection(set(anomaly_id_list)))
        res = res/c
        _dict[n] = res

    x = list(_dict.keys())
    y = list(_dict.values())
    print(x,y)
    return x,y
