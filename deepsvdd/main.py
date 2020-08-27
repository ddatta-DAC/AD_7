import torch
import random
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
sys.path.append('../..')
sys.path.append('../')
try:
    from .networks.AE import FC_dec
    from .networks.AE import FC_enc
except:
    from networks.AE import FC_dec
    from networks.AE import FC_enc

from torch import FloatTensor as FT
from torch import LongTensor as LT
from torch import nn
from torch.nn import functional as F
import os
from collections import OrderedDict
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from sklearn.cluster import MiniBatchKMeans, KMeans
import argparse
try:
    from . import utils
except:
    import utils
try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher

try:
    from deepSVDD import DeepSVDD
except:
    from .deepSVDD import DeepSVDD

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device ::', DEVICE)

################################################################################
# Settings
################################################################################

def main(
        data_dict,
        layer_dims,
        objective='soft-boundary',
        config=None,
        NU = None
):
    global DEVICE
    LR = config['LR']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    warm_up_epochs = config['warm_up_epochs']
    ae_epochs = config['ae_epochs']
    num_anomaly_sets = config['num_anomaly_sets']
    train_X = data_dict['train'].values
    fc_layer_dims = [train_X.shape[1]] + list(layer_dims)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(
        DEVICE,
        objective=objective,
        nu = NU
    )
    deep_SVDD.set_network(fc_layer_dims)

    # Train model on dataset
    deep_SVDD.train(
        train_X,
        LR = LR,
        num_epochs = num_epochs,
        batch_size= batch_size,
        ae_epochs = ae_epochs,
        warm_up_epochs=warm_up_epochs
    )

    # Test model

    test_X = data_dict['test'].values
    test_scores = deep_SVDD.test(test_X)
    test_labels = [0 for _ in range(test_X.shape[0])]
    auc_list = []

    for idx in range(num_anomaly_sets):
        key = 'anom_' + str(idx + 1)
        anom_X = data_dict[key].values
        anom_labels = [1 for _ in range(anom_X.shape[0])]
        anom_scores = deep_SVDD.test(anom_X)

        combined_scores = np.concatenate([anom_scores, test_scores], axis=0)
        combined_labels = np.concatenate([anom_labels, test_labels], axis=0)

        res_data = []
        for i, j in zip(combined_scores, combined_labels):
            res_data.append((i, j))
        res_df = pd.DataFrame(res_data, columns=['score', 'label'])

        #  Normalize values
        def _normalize_(val, _min, _max):
            return (val - _min) / (_max - _min)

        _max = max(combined_scores)
        _min = min(combined_scores)

        res_df['score'] = res_df['score'].parallel_apply(
            _normalize_,
            args=(_min, _max,)
        )

        res_df = res_df.sort_values(by=['score'], ascending=False)
        _max = max(res_df['score'])
        _min = min(res_df['score'])
        step = (_max - _min) / 100

        # Vary the threshold
        thresh = _max - step
        num_anomalies = anom_X.shape[0]
        P = []
        R = [0]

        while thresh >= _min:
            sel = res_df.loc[res_df['score'] >= thresh]
            if len(sel) == 0:
                thresh -= step
                continue
            correct = sel.loc[sel['label'] == 1]
            prec = len(correct) / len(sel)
            rec = len(correct) / num_anomalies
            P.append(prec)
            R.append(rec)
            if rec >= 1.0:
                break
            thresh -= step
            thresh = round(thresh, 3)
        P = [P[0]] + P
        from sklearn.metrics import auc

        pr_auc = auc(R, P)
        auc_list.append(pr_auc)

        print("AUC : {:0.4f} ".format(pr_auc))
        try:
            plt.figure()
            plt.title('PR Curve' + str(pr_auc))
            plt.plot(R, P)
            plt.show()
        except:
            pass

    _mean = np.mean(auc_list)
    _std = np.std(auc_list)
    print(' Mean AUC ', np.mean(auc_list))
    print(' AUC std', np.std(auc_list))
    return _mean, _std

# ==================================================

parser = argparse.ArgumentParser(description='Run the model ')
parser.add_argument(
    '--DATA_SET',
    type=str,
    help=' Which data set ?',
    default=None,
    choices=['kddcup', 'kddcup_neptune', 'nsl_kdd', 'nb15','gureKDD']
)

parser.add_argument(
    '--num_runs',
    type=int,
    default=5,
    help='Number of runs'
)

args = parser.parse_args()
DATA_SET = args.DATA_SET
num_runs = args.num_runs
LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = utils.get_logger(LOG_FILE)
utils.log_time(LOGGER)
LOGGER.info(DATA_SET)
config_file = 'config.yaml'
with open(config_file, 'r') as fh:
    config = yaml.safe_load(fh)

num_anomaly_sets = config[DATA_SET]['num_anomaly_sets']
anomaly_ratio = config[DATA_SET]['anomaly_ratio']
config = config[DATA_SET]
print(config)
layer_dims = config['layer_dims']

# =================================
# Vary NU
# =================================
NU_values = [0.01,0.05,0.1]
objectives =['soft-boundary','one-class']
LOGGER.info(str(config))

for _NU in NU_values:
    LOGGER.info('Setting NU to {}'.format(_NU))
    results_sb = []
    results_oc =[]
    for n in range(1,num_runs+1):
        data_dict, _ = data_fetcher.get_data(
            DATA_SET,
            one_hot=True,
            num_anom_sets=num_anomaly_sets,
            anomaly_ratio=anomaly_ratio
        )
        mean_aupr1, std1 = main(
            data_dict,
            layer_dims,
            objective='soft-boundary',
            config=config,
            NU=_NU
        )
        results_sb.append(mean_aupr1)
        mean_aupr2, std2 = main(
            data_dict,
            layer_dims,
            objective='one-class',
            config=config,
            NU=_NU
        )
        results_oc.append(mean_aupr2)
        LOGGER.info(' Run {}: Objective {} Mean: {:4f} | Std {:4f} || Objective {} Mean: {:4f} | Std {:4f}'.format(
            n,'soft-boundary', mean_aupr1, std1, 'one-class' , mean_aupr2, std2))
        
    mean_all_runs_1 = np.mean(results_sb)
    mean_all_runs_2 = np.mean(results_oc)
    print('Mean AuPR over {} runs Objective {} {:4f}'.format(num_runs, 'soft-boundary', mean_all_runs_1))
    print('Mean AuPR over {} runs Objective {} {:4f}'.format(num_runs, 'one-class',     mean_all_runs_2))
    
    LOGGER.info('Mean AuPR over {} runs Objective {} {:4f}  Std {:4f}'.format(num_runs, 'soft-boundary', mean_all_runs_1,  np.std(results_sb)))
    LOGGER.info('Mean AuPR over {} runs Objective {} {:4f}  Std {:4f}'.format(num_runs, 'one-class',     mean_all_runs_2,  np.std(results_oc)))
    
#     LOGGER.info('Mean AuPR over  {} runs {:4f} Std {:4f}'.format(num_runs, mean_all_runs, np.std(results)))
    LOGGER.info(' Details ' + str(results_sb))
    LOGGER.info(' Details ' + str(results_oc))
    
utils.close_logger(LOGGER)





