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
from joblib import Parallel,delayed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
# ===============================================
try:
    from model import model_9_container as Model
except:
    from .model import model_9_container as Model

try:
    from . import utils as utils
except:
    import utils as utils
try:
    from . import model_data_fetcher
except:
    import model_data_fetcher as model_data_fetcher


def execute_run(
        DATA_SET,
        set_id = 1,
        show_figure = False
):
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

    num_anomaly_sets = 5
    pos, neg, data_dict = model_data_fetcher.fetch_model_data(
        DATA_SET,
        set_id=set_id,
        num_anom_sets=num_anomaly_sets,
        anomaly_ratio=anomaly_ratio
    )

    not_converged = True

    # This is a simple check for convergence
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
        print('Network architecture ', ae_model.network_module)
        _, epoch_losses_phase_3 = ae_model.train_model(
            pos,
            neg
        )

        if epoch_losses_phase_3[-1] < epoch_losses_phase_3[0]:
            not_converged = False

    ae_model.mode = 'test'
    auc_list = utils.evaluate(ae_model,data_dict,num_anomaly_sets, show_figure)
    _mean = np.mean(auc_list)
    _std = np.std(auc_list)
    print(' Mean AUC {:0.4f} '.format(_mean))
    print(' AUC std {:0.4f} '.format(_std))
    return _mean,_std,set_id

# ==========================================================

parser = argparse.ArgumentParser(description='Run the model ')

parser.add_argument(
    '--DATA_SET',
    type=str,
    help=' Which data set ?',
    default='kddcup',
    choices=['kddcup']
)

parser.add_argument(
    '--show_figure',
    type=bool,
    help=' Show AuPR curve ?',
    default=False
)

parser.add_argument(
    '--demo',
    type=bool,
    help=' Show AuPR curve ?',
    default=True
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
show_figure = args.show_figure
demo = args.demo
if demo:
    num_runs = 1
    show_figure = True

LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = utils.get_logger(LOG_FILE)

utils.log_time(LOGGER)
LOGGER.info(DATA_SET)
results = []

if num_runs > 1:
    run_ids = range(1,num_runs+1)
else:
    run_ids = [np.random.randint(1,10+1)]

for n in run_ids:
    mean_aupr, std , _id = execute_run(DATA_SET, n, show_figure)
    results.append(mean_aupr)
    LOGGER.info('  Mean: {:4f} | Std {:4f}'.format( mean_aupr,std))
mean_all_runs = np.mean(results)
std_all_runs = np.std(results)

# all_results = Parallel(n_jobs=5)(delayed(execute_run)(DATA_SET, n) for n in range(1,num_runs+1))
# all_results = np.array(all_results)
# for n in range(1,num_runs+1):
#     mean_aupr = all_results[n-1][0] 
#     std = all_results[n-1][1]
#     _id = all_results[n-1][2]
#     results.append(mean_aupr)
#     LOGGER.info(' Run {}: Mean: {:4f} | Std {:4f}'.format(_id, mean_aupr, std))
# mean_all_runs = np.mean(all_results[:,0])
# std_all_runs = np.std(all_results[:,0])
# results = all_results[:,0]

print('Mean AuPR over  {} runs {:4f}'.format(num_runs, mean_all_runs))
print('Details: ', str(results))

LOGGER.info('Mean AuPR over  {} runs {:4f} Std {:4f}'.format(num_runs, mean_all_runs, std_all_runs))
LOGGER.info(' Details ' + str(results))
utils.close_logger(LOGGER)
