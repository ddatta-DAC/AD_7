#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import sys
import os
sys.path.append('./..')
sys.path.append('./../..')
import pandas as pd
import yaml
from torch import FloatTensor as FT
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch
from pprint import pprint
from collections import OrderedDict

try:
    from eval import eval
except:
    from .eval import eval
try:
    from . import logger_utils
except:
    import logger_utils
try:
    from data_fetcher_v2 import data_fetcher
except:
    from .data_fetcher_v2 import data_fetcher
import argparse
from pathlib import Path
import yaml
from sklearn.metrics import auc
from sklearn.svm import OneClassSVM as OCSVM



def train_model(
        data_dict,
        nu
    ):
    model_obj = OCSVM(
        kernel='rbf',
        nu = nu,
        shrinking=True,
        cache_size=1000,
        verbose=True,
        max_iter=-1
    )
    train_X = data_dict['train']
    model_obj.fit(train_X)
    return model_obj

def test_eval(model_obj, data_dict, num_anomaly_sets):
    test_X = data_dict['test']
    test_scores = model_obj.score_samples(test_X)
    auc_list = []
    for idx in range(num_anomaly_sets):
        key = 'anom_' + str(idx + 1)
        anom_X = data_dict[key]
        anom_scores = model_obj.score_samples(anom_X)
        auPR = eval.eval(anom_scores, test_scores, order='ascending')
        auc_list.append(auPR)
        print("AUC : {:0.4f} ".format(auPR))
    _mean = np.mean(auc_list)
    _std = np.std(auc_list)
    print(' Mean AUC ', np.mean(auc_list))
    print(' AUC std', np.std(auc_list))
    return _mean, _std

# ==============================================================
parser = argparse.ArgumentParser(description='Run the model ')
parser.add_argument(
    '--DATA_SET',
    type=str,
    help=' Which data set ?',
    default='kddcup',
    choices=['kddcup', 'kddcup_neptune', 'nsl_kdd', 'nb15','gureKDD']
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=1,
    help='Number of runs'
)

# =========================================
args = parser.parse_args()
DATA_SET = args.DATA_SET
num_runs = args.num_runs
LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = logger_utils.get_logger('OCSVM')
DATA_SET(LOGGER)

LOGGER.info(DATA_SET)
config_file = 'config.yaml'
with open(config_file, 'r') as fh:
    config = yaml.safe_load(fh)

num_anomaly_sets = config[DATA_SET]['num_anomaly_sets']
anomaly_ratio = config[DATA_SET]['anomaly_ratio']

anom_perc = 100 * anomaly_ratio/(1+anomaly_ratio)
nu_values = np.arange(0.1,0.5+0.1,0.10)
nu_vs_auc = []
for nu in nu_values:
    results = []
    LOGGER.info('Setting nu :: {}'.format(nu))
    for n in range(1,num_runs+1):
        data_dict, _ = data_fetcher.get_data(
            DATA_SET,
            set_id = n,
            num_anom_sets=num_anomaly_sets,
            anomaly_perc=anom_perc
        )
        model_obj = train_model(DATA_SET, data_dict, config)
        mean_aupr, std = test_eval(model_obj, data_dict, num_anomaly_sets)
        results.append(mean_aupr)
        LOGGER.info(' Run {}: Mean: {:4f} | Std {:4f}'.format(n,mean_aupr,std))
        mean_all_runs = np.mean(results)
        print('Mean AuPR over {} runs {:4f}'.format(num_runs, mean_all_runs))
        print('Details: ', results)
        nu_vs_auc.append((nu, mean_all_runs))

nu_vs_auc = np.array(nu_vs_auc)
LOGGER.info('nu vs AuPR '+ str(nu_vs_auc[:,0]), str(nu_vs_auc[:,1]))
logger_utils.close_logger(LOGGER)

