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
from joblib import Parallel,delayed
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
try:
    from .DCN_1.model_dcn import DCN
except:
    from DCN_1.model_dcn import DCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device ::', DEVICE)


import argparse
from pathlib import Path
import yaml
from sklearn.metrics import auc
from sklearn.svm import OneClassSVM as OCSVM

def train_model(
        data_dict,
        config,
        K
    ):
    global DEVICE
    data_dim = data_dict['train'].shape[1]
    layer_dims = config['layer_dims']
    epochs_1 = config['epochs_1']
    epochs_2 = config['epochs_2']
    batch_size = config['batch_size']
    layer_dims = config['layer_dims']
    model_obj = DCN(
        DEVICE,
        data_dim,
        layer_dims,  # Provide the half (encoder only)
        op_activation='sigmoid',
        layer_activation='sigmoid',
        dropout=0.1,
        LR=0.001,
        num_epochs_1=epochs_1,
        num_epochs_2=epochs_2,
        min_epochs=5,
        batch_size=batch_size,
        k=K,
        stop_threshold=0.05,
        checkpoint_dir=DATA_SET,
    )
    train_X = data_dict['train']
    model_obj.train_model(train_X)
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


def execute(DATA_SET, id, K, config, anom_perc, num_anomaly_sets ):
    data_dict, _ = data_fetcher.get_data(
        DATA_SET,
        set_id=id,
        num_anom_sets=num_anomaly_sets,
        anomaly_perc=anom_perc
    )

    model_obj = train_model(
        data_dict,
        config,
        K
    )
    mean_aupr, std = test_eval(model_obj, data_dict, num_anomaly_sets)
    return (mean_aupr, std)
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
    default=10,
    help='Number of runs'
)

# =========================================

args = parser.parse_args()
DATA_SET = args.DATA_SET
num_runs = args.num_runs
LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = logger_utils.get_logger(LOG_FILE,'DCN')

LOGGER.info(DATA_SET)
config_file = 'config.yaml'
with open(config_file, 'r') as fh:
    config = yaml.safe_load(fh)

num_anomaly_sets = config[DATA_SET]['num_anomaly_sets']
anomaly_ratio = config[DATA_SET]['anomaly_ratio']
model_config = config[DATA_SET]['dcn']

anom_perc = 100 * anomaly_ratio/(1+anomaly_ratio)
step=1
K_values = np.arange(1,10+step,step)
nu_vs_auc = []
for K in K_values:
    K = int(K)
    LOGGER.info('Setting nu :: {}'.format(nu))
    _res_ = Parallel(n_jobs=num_runs)(delayed(execute)(
        DATA_SET, id, K, model_config, anom_perc, num_anomaly_sets ) for id in range(1,num_runs+1)
    )

    results = np.array(_res_)
    mean_all_runs = np.mean(results[:,0])
    _std = np.std(results[:, 0])
    LOGGER.info(' Runs {}: Mean: {:4f} | Std {:4f}'.format(num_runs, mean_all_runs, _std))
    print('Mean AuPR over {} runs {:4f}'.format(num_runs, mean_all_runs))
    print('Details: ', results[:,0])
    nu_vs_auc.append((nu, mean_all_runs))

nu_vs_auc = np.array(nu_vs_auc)
LOGGER.info('nu vs AuPR '+ str(nu_vs_auc[:,0]) +  str(nu_vs_auc[:,1]))
logger_utils.close_logger(LOGGER)

