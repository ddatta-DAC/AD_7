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
    from model_9.model import model_9_container as Model
except:
    from .model_9.model import model_9_container as Model

try:
    from model_6 import utils as utils
except:
    from .model_6 import utils as utils

try:
    from model_6 import model_data_fetcher
except:
    from .model_6 import model_data_fetcher as model_data_fetcher


def execute_run(
        DATA_SET,
        pos,
        neg,
        data_dict,
        config
):
    global LOGGER
    encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim = utils.create_config(
        DATA_SET)
    anomaly_ratio = -1
    ae_model = None

    burn_in_epochs = config['burn_in_epochs']
    phase_2_epochs = config['phase_2_epochs']
    phase_3_epochs = config['phase_3_epochs']
    batch_size = config['batch_size']
    ae_dropout = config['ae_dropout']
    fc_dropout = config['fc_dropout']

    LR = config['LR']
    max_gamma = config['max_gamma']

    # ===============
    # 1. Train with noise
    # ===============
    not_converged = True
    while not_converged:
        ae_model = Model(
            DATA_SET,
            DEVICE,
            latent_dim,
            encoder_structure_config,
            decoder_structure_config,
            loss_structure_config,
            batch_size=batch_size,
            fc_dropout=fc_dropout,
            ae_dropout=ae_dropout,
            learning_rate=LR,
            max_gamma=max_gamma,
            burn_in_epochs=burn_in_epochs,
            phase_2_epochs=phase_2_epochs,
            phase_3_epochs=phase_3_epochs,
            include_noise=True
        )

        _, epoch_losses_phase_3 = ae_model.train_model(
            pos,
            neg
        )
        print(epoch_losses_phase_3)
        if epoch_losses_phase_3[-1] < epoch_losses_phase_3[0]:
            not_converged = False

        if DATA_SET == 'nb15' and epoch_losses_phase_3[-1] >= 0.1:
            not_converged = True

    test_norm_X = data_dict['test']
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
        res_df = res_df.sort_values(by=['score'], ascending=True)

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
        thresh = _min + step
        thresh = round(thresh, 3)
        num_anomalies = x2.shape[0]
        print('Num anomalies', num_anomalies)
        P = []
        R = [0]

        while thresh <= _max + step:

            sel = res_df.loc[res_df['score'] <= thresh]
            if len(sel) == 0:
                thresh += step
                continue

            correct = sel.loc[sel['label'] == 0]
            prec = len(correct) / len(sel)
            rec = len(correct) / num_anomalies
            P.append(prec)
            R.append(rec)
            thresh += step
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
    return _mean, _std


# ==========================================================

parser = argparse.ArgumentParser(description='Run the model ')
parser.add_argument(
    '--DATA_SET',
    type=str,
    help=' Which data set ?',
    default='kddcup',
    choices=['kddcup', 'kddcup_neptune', 'nsl_kdd', 'nb15', 'gureKDD']
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=5,
    help='Number of negative samples'
)


parser.add_argument(
    '--neg_samples_min',
    type=int,
    default=2,
    help='Number of negative samples'
)

parser.add_argument(
    '--neg_samples_max',
    type=int,
    default=20,
    help='Number of negative samples'
)

parser.add_argument(
    '--neg_samples_step',
    type=int,
    default=2,
    help='Number of negative samples'
)

# ====================================== #

args = parser.parse_args()
DATA_SET = args.DATA_SET

neg_samples_min = args.neg_samples_min
neg_samples_max = args.neg_samples_max
neg_samples_step = args.neg_samples_step
num_runs = args.num_runs

LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = utils.get_logger(LOG_FILE)
utils.log_time(LOGGER)
LOGGER.info(DATA_SET)
LOGGER.info(' Varying negative samples {} -- {}, step {}'.format(neg_samples_min, neg_samples_max, neg_samples_step))
results = []


# ================================================================

def format_arr(arr):
    res = ''
    for a in arr:
        res += ', {:4f}'.format(a)
    return res


config_file = 'architecture_config.yaml'
with open(config_file, 'r') as fh:
    config = yaml.safe_load(fh)
config = config[DATA_SET]
anomaly_ratio = config['anomaly_ratio']
num_anomaly_sets = 5

# =======================================================================================
# On the same data, run the model multiple times , varying the number of negative samples
# =======================================================================================

for nr in range(num_runs):
    pos, neg, data_dict = model_data_fetcher.fetch_model_data(
        DATA_SET,
        num_anom_sets=num_anomaly_sets,
        anomaly_ratio=anomaly_ratio,
        num_neg_samples=neg_samples_max
    )
    runs_aupr = []
    for neg_samples in range(neg_samples_min , neg_samples_max+neg_samples_step, neg_samples_step):
        _neg = neg[:,:neg_samples,:]
        _mean_aupr, std = execute_run(
            DATA_SET,
            pos,
            _neg,
            data_dict,
            config
        )
        runs_aupr.append(_mean_aupr)

    _mean = np.mean(runs_aupr)
    LOGGER.info('Run {} : AUPR Mean : {:4f} '.format(nr+1 , _mean))
    LOGGER.info(format_arr(runs_aupr))
    results.append(runs_aupr)
# ================================

results = np.array(results)
idx = 0
for neg_samples in range(neg_samples_min , neg_samples_max+neg_samples_step, neg_samples_step):
    arr = results[:,idx]
    _mean = np.mean(arr)
    _std = np.std(arr)
    LOGGER.info('Neg Samples  {}: AUPR Mean: {:4f} Std: {:4f}'.format(neg_samples, _mean, _std))
    idx +=1

utils.close_logger(LOGGER)

