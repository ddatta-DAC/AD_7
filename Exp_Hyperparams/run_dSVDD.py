import torch
import random
import numpy as np
import os
import sys
import pandas as pd
sys.path.append('../../../.')
sys.path.append('../../.')
sys.path.append('../')
import yaml
from tqdm import tqdm
import argparse
from joblib import Parallel, delayed

try:
    from .deepsvdd.networks.AE import FC_dec
    from .deepsvdd..AE import FC_enc
    from .deepsvdd.deepSVDD import DeepSVDD
except:
    from deepsvdd.networks.AE import FC_dec
    from deepsvdd.networks.AE import FC_enc
    from deepsvdd.deepSVDD import DeepSVDD

try:
    from eval import eval
except:
    from .eval import eval
try:
    from . import logger_utils
except:
    import logger_utils
try:
    from .data_fetcher_v2 import data_fetcher
except:
    from data_fetcher_v2 import data_fetcher


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device ::', DEVICE)

def train_model(
        data_dict,
        config,
        objective='soft-boundary',
        nu = 0.01
):
    global DEVICE
    layer_dims = config['layer_dims']
    LR = config['LR']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    warm_up_epochs = config['warm_up_epochs']
    ae_epochs = config['ae_epochs']
    train_X = data_dict['train']
    fc_layer_dims = [train_X.shape[1]] + list(layer_dims)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(
        DEVICE,
        objective=objective,
        nu = nu
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
    return deep_SVDD


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



def execute(DATA_SET, nu, objective,  id , config, anom_perc, num_anomaly_sets ):
    data_dict, _ = data_fetcher.get_data(
        DATA_SET,
        set_id=id,
        num_anom_sets=num_anomaly_sets,
        anomaly_perc=anom_perc
    )
    model_obj = train_model(data_dict, config = config, nu=nu, objective = objective)
    mean_aupr, std = test_eval(model_obj, data_dict, num_anomaly_sets)
    return (mean_aupr, std)

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
LOGGER = logger_utils.get_logger(LOG_FILE,'deepSVDD')
LOGGER.info(DATA_SET)
config_file = 'config.yaml'

with open(config_file, 'r') as fh:
    config = yaml.safe_load(fh)

num_anomaly_sets = config[DATA_SET]['num_anomaly_sets']
anomaly_ratio = config[DATA_SET]['anomaly_ratio']
anom_perc = 100 * anomaly_ratio/(1+anomaly_ratio)
step = 0.025
nu_values = np.arange(0.025,0.2+step,step)
nu_vs_auc = []
objective = 'one-class'

model_config = config[DATA_SET]
for nu in nu_values:
    LOGGER.info('Setting nu :: {}'.format(nu))
    _res_ = Parallel(n_jobs=num_runs)(delayed(execute)(
        DATA_SET, nu, objective, id, model_config, anom_perc, num_anomaly_sets ) for id in range(1,num_runs+1)
    )
    results = np.array(_res_)
    mean_all_runs = np.mean(results[:,0])
    _std = np.std(results[:,0])
    LOGGER.info(' Runs {}: Mean: {:4f} | Std {:4f}'.format(num_runs, mean_all_runs, _std))
    print('Mean AuPR over {} runs {:4f}'.format(num_runs, mean_all_runs))
    print('Details: ', results[:,0])
    nu_vs_auc.append((nu, mean_all_runs))

nu_vs_auc = np.array(nu_vs_auc)
LOGGER.info('nu vs AuPR '+ str(nu_vs_auc[:,0]) + str(nu_vs_auc[:,1]))
logger_utils.close_logger(LOGGER)
