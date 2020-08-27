import sys
import os
import pandas as pd
import numpy as np

sys.path.append('./..')
sys.path.append('./../..')
import torch
import math
import yaml
from sklearn.metrics import auc
from tqdm import tqdm
from collections import OrderedDict
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
import multiprocessing
from pprint import pprint
from time import time
from datetime import datetime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
# ===============================================
try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher
try:
    from model_aef_r import model_FAER_container as Model
except:
    from .model_aef_r import model_FAER_container as Model
try:
    from utils import create_config
except:
    from .utils import create_config
try:
    import utils
except:
    from . import utils



def execute_run(DATA_SET):

    global LOGGER
    encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim = create_config(DATA_SET)
    anomaly_ratio = -1
    ae_model = None
    config_file = 'architecture_config.yaml'

    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)

    anomaly_ratio = config[DATA_SET]['anomaly_ratio']
    LR = config[DATA_SET]['LR']
    batch_size =  config[DATA_SET]['batch_size']
    epochs = config[DATA_SET]['epochs']
    dropout = config[DATA_SET]['ae_dropout']

    ae_model = Model(
        DEVICE,
        latent_dim,
        encoder_structure_config,
        decoder_structure_config,
        loss_structure_config,
        optimizer='Adam',
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=LR,
        dropout=dropout
    )

    print(ae_model.network_module)

    num_anomaly_sets = 5
    data_dict, _ = data_fetcher.get_data(
        DATA_SET,
        one_hot=True,
        num_anom_sets= num_anomaly_sets,
        anomaly_ratio= anomaly_ratio
    )

    train_df = data_dict['train']
    train_X = train_df.values
    ae_model.train_model(
        train_X
    )
    test_norm_df = data_dict['test']
    test_norm_X = test_norm_df.values

    auc_list = []
    ae_model.mode = 'test'

    def _normalize_(val, _min, _max):
        return (val - _min) / (_max - _min)

    for idx in range(1, num_anomaly_sets + 1):
        key = 'anom_' + str(idx)
        test_anom_df = data_dict[key]
        test_anom_X = test_anom_df.values
        x1 = test_norm_X
        x2 = test_anom_X

        x1_scores = ae_model.get_score(x1)
        x2_scores = ae_model.get_score(x2)

        res_data = []
        labels = [1 for _ in range(x1.shape[0])] + [0 for _ in range(x2.shape[0])]
        _scores = np.concatenate([x1_scores, x2_scores], axis=0)
        for i, j in zip(_scores, labels):
            res_data.append((i[0], j))

        res_df = pd.DataFrame(res_data, columns=['score', 'label'])
        res_df = res_df.sort_values(by=['score'], ascending=False)

        _max = max(res_df['score'])
        _min = min(res_df['score'])

        res_df['score'] = res_df['score'].parallel_apply(
            _normalize_,
            args=(_min, _max,)
        )

        _max = max(res_df['score'])
        _min = min(res_df['score'])

        step = (_max - _min) / 100

        # Vary the threshold
        thresh = _max - step
        thresh = round(thresh, 3)
        num_anomalies = x2.shape[0]
        P = []
        R = [0]
        while thresh >= _min :
            sel = res_df.loc[res_df['score'] >= thresh]
            if len(sel) == 0:
                thresh -= step
                continue

            correct = sel.loc[sel['label'] == 0]

            prec = len(correct) / len(sel)
            rec = len(correct) / num_anomalies
            P.append(prec)
            R.append(rec)
            thresh -= step
            thresh = round(thresh, 3)

        P = [P[0]] + P
        pr_auc = auc(R, P)
        try:
            plt.figure(figsize=[8, 6])
            plt.plot(R, P)
            plt.title('Precision Recall Curve  || auPR :' + "{:0.4f}".format(pr_auc), fontsize=15)
            plt.xlabel('Recall', fontsize=15)
            plt.ylabel('Precision', fontsize=15)
            plt.show()
        except:
            pass
        print("AUC : {:0.4f} ".format(pr_auc))
        auc_list.append(pr_auc)
    _mean = np.mean(auc_list)
    _std = np.std(auc_list)
    print(' Mean AUC {:0.4f} '.format(_mean))
    print(' AUC std {:0.4f} '.format(_std))
    return _mean,_std

# ==========================================================

parser = argparse.ArgumentParser(description='Run the model ')
parser.add_argument(
    '--DATA_SET',
    type=str,
    help=' Which data set ?',
    default=None,
    choices=['kddcup', 'kddcup_neptune', 'nsl_kdd', 'nb15', 'gureKDD']
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=10,
    help='Number of runs'
)

args = parser.parse_args()
DATA_SET = args.DATA_SET
num_runs = args.num_runs
LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = utils.get_logger(LOG_FILE)
utils.log_time(LOGGER)
LOGGER.info(DATA_SET)
results = []
for n in range(1,num_runs+1):
    mean_aupr, std = execute_run(DATA_SET)
    results.append(mean_aupr)
    LOGGER.info(' Run {}: Mean: {:4f} | Std {:4f}'.format(n,mean_aupr,std))
mean_all_runs = np.mean(results)
print('Mean AuPR over  {} runs {:4f}'.format(num_runs, mean_all_runs))
print('Details: ', results)

LOGGER.info('Mean AuPR over  {} runs {:4f} Std {:4f}'.format(num_runs, mean_all_runs, np.std(results)))
LOGGER.info(' Details ' + str(results))

utils.close_logger(LOGGER)
