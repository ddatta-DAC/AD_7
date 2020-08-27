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
    from model_7.model import model_6_v2_container as Model
except:
    from .model_7.model import model_6_v2_container as Model

try:
    from model_6 import utils as utils
except:
    from .model_6 import utils as utils

try:
    from model_6 import model_data_fetcher
except:
    from .model_6 import model_data_fetcher as model_data_fetcher


def execute_run(DATA_SET):
    global LOGGER
    encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim = utils.create_config(DATA_SET)
    anomaly_ratio = -1
    ae_model = None

    config_file = 'architecture_config.yaml'

    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    config = config[DATA_SET]

    burn_in_epochs = config['burn_in_epochs']
    phase_2_epochs = config['phase_2_epochs']
    phase_3_epochs = config['phase_3_epochs']
    batch_size = config['batch_size']
    ae_dropout = config['ae_dropout']
    fc_dropout = config['fc_dropout']
    anomaly_ratio = config['anomaly_ratio']
    LR = config['LR']
    max_gamma = config['max_gamma']

    # Get a single set
    num_anomaly_sets = 1
    pos, neg, data_dict = model_data_fetcher.fetch_model_data(
        DATA_SET,
        num_anom_sets=num_anomaly_sets,
        anomaly_ratio=anomaly_ratio
    )

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
            phase_3_epochs=phase_3_epochs
        )

        print(ae_model.network_module)

        _, epoch_losses_phase_3 = ae_model.train_model(
            pos,
            neg
        )
        print(epoch_losses_phase_3)
        if epoch_losses_phase_3[-1] < epoch_losses_phase_3[0]:
            not_converged = False
            break

    test_norm_X = data_dict['test']
    auc_list = []
    ae_model.mode = 'test'

    def _normalize_(val, _min, _max):
        return (val - _min) / (_max - _min)

    anom_perc_values = [2,4,6,8,10]

    # Vary the percentage of anomalies
    for perc_value in anom_perc_values :

        key = 'anom_' + str(1)
        test_anom_df = data_dict[key]
        calculated_num_anom = int(float(perc_value)/(100-perc_value) * test_norm_X.shape[0])
        _test_anom_df = test_anom_df.sample(n=calculated_num_anom)
        test_anom_X = _test_anom_df.values

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
        LOGGER.info('Percentage of anomalies {}  AUC: {:0.4f}  '.format(perc_value,pr_auc))

    return auc_list

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
    auc_list = execute_run(DATA_SET)
    print('AUC list : ', auc_list)
    LOGGER.info(' Run {}:'.format(n) + str(auc_list) )
    results.append(auc_list)

# ================================== #
results = np.array(results)

for i in range(results.shape[1]):
    _mean = np.mean(results[:,i])
    _std = np.std(results[:,i])
    print(' Point {}'.format(i+1), _mean)
    LOGGER.info('Mean at Point {} : Mean  {:4f}  Std {:4f}'.format(i+1, _mean, _std) )


utils.close_logger(LOGGER)
