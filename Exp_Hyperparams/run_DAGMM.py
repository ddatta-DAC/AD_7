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
from torch import FloatTensor as FT
from torch.autograd import Variable
import torch
from pprint import pprint
from collections import OrderedDict
from joblib import Parallel,delayed
import argparse
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device ::', DEVICE)
GPU_COUNT = torch.cuda.device_count()

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
    from .DAGMM.base_DAGMM_v2 import DaGMM
except:
    from DAGMM.base_DAGMM_v2 import DaGMM

def create_config(
        data_dim,
        config,
        K
):
    latent_dim = config['ae_latent_dimension']
    encoder_structure_config = {}
    encoder_structure_config['num_discrete'] = data_dim
    encoder_structure_config['num_real'] = 0
    encoder_structure_config['discrete_column_dims'] = None
    encoder_structure_config['encoder_layers'] = {
        'activation': config['encoder_layers']['activation'],
        'layer_dims': config['encoder_layers']['layer_dims'] + [latent_dim]
    }


    # ======================================================
    # Set decoder structure
    # =========

    decoder_structure_config = {}
    final_op_dims = data_dim
    decoder_structure_config['discrete_column_dims'] = None

    decoder_structure_config['decoder_layers'] = {
        'activation': config['decoder_layers']['activation'],
        'layer_dims': [latent_dim] + config['decoder_layers']['layer_dims'] + [final_op_dims]
    }
    decoder_structure_config['final_output_dim'] = final_op_dims
    # =====================
    # GMM
    # =====================
    gmm_input_dims = latent_dim + 2
    activation = config['gmm']['FC_layer']['activation']
    num_components = config['gmm']['num_components']
    FC_layer_dims = [gmm_input_dims] + config['gmm']['FC_layer']['dims'] + [num_components]
    FC_dropout = config['gmm']['FC_dropout']
    gmm_structure_config = {
        'num_components': K,
        'FC_layer_dims': FC_layer_dims,
        'FC_dropout': FC_dropout,
        'FC_activation': activation
    }
    loss_structure_config = []
    return encoder_structure_config, decoder_structure_config, gmm_structure_config, loss_structure_config, latent_dim

def train_model(
        dagmm_obj,
        data,
        _DEVICE,
        num_epochs=400,
        batch_size=512,
        LR=0.001,
):
    optimizer = torch.optim.Adam(dagmm_obj.parameters(), lr=LR)
    dagmm_obj.train()
    log_interval = 100
    for epoch in tqdm(range(num_epochs)):
        num_batches = data.shape[0] // batch_size + 1
        epoch_losses = []
        np.random.shuffle(data)
        # X = FT(data).to(DEVICE)
        X = data
        lambda_energy = 0.1
        lambda_cov_diag = 0.005
        for b in range(num_batches):
            optimizer.zero_grad()
            input_data = X[b * batch_size: (b + 1) * batch_size]
            input_data = FT(input_data).to(_DEVICE)
            enc, dec, z, gamma = dagmm_obj(input_data)
            total_loss, sample_energy, recon_error, cov_diag = dagmm_obj.loss_function(
                input_data, dec, z, gamma,
                lambda_energy,
                lambda_cov_diag
            )

            dagmm_obj.zero_grad()
            total_loss = Variable(total_loss, requires_grad=True)
            total_loss.backward()
            epoch_losses.append(total_loss.cpu().data.numpy())
            torch.nn.utils.clip_grad_norm_(dagmm_obj.parameters(), 5)
            optimizer.step()

            loss = {}
            loss['total_loss'] = total_loss.data.item()
            loss['sample_energy'] = sample_energy.item()
            loss['recon_error'] = recon_error.item()
            loss['cov_diag'] = cov_diag.item()

            if (b + 1) % log_interval == 0:
                log = ' '
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
        print('Epoch loss ::', np.mean(epoch_losses))
    return dagmm_obj


def get_model_obj(
        data_dict,
        model_config,
        K,
        _DEVICE
):

    batch_size = model_config['batch_size']
    num_epochs = model_config['num_epochs']
    train_X = data_dict['train']
    data_dim = train_X.shape[1]
    encoder_structure_config, decoder_structure_config, gmm_structure_config, _, latent_dim = create_config(
        data_dim,
        model_config,
        K
    )

    dagmm_obj = DaGMM(
        _DEVICE,
        encoder_structure_config,
        decoder_structure_config,
        n_gmm=gmm_structure_config['num_components'],
        ae_latent_dim=latent_dim
    )
    dagmm_obj = dagmm_obj.to(_DEVICE)
    print(dagmm_obj)
    dagmm_obj = train_model(
        dagmm_obj,
        train_X,
        _DEVICE,
        num_epochs=num_epochs,
        batch_size=batch_size,
        LR=0.0001
    )
    return dagmm_obj

def test_eval(
    dagmm_obj,
    data_dict,
    num_anom_sets,
    _DEVICE
):
    dagmm_obj.eval()
    N = 0
    mu_sum = 0
    cov_sum = 0
    gamma_sum = 0

    train_X = data_dict['train']
    batch_size = 507
    num_batches = train_X.shape[0] // batch_size + 1
    for b in range(num_batches):
        input_data = train_X[b * batch_size: (b + 1) * batch_size]
        input_data = FT(input_data).to(_DEVICE)
        enc, dec, z, gamma = dagmm_obj(input_data)
        phi, mu, cov = dagmm_obj.compute_gmm_params(z, gamma)
        batch_gamma_sum = torch.sum(gamma, dim=0)
        gamma_sum += batch_gamma_sum
        mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
        cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only
        N += input_data.size(0)

    train_phi = gamma_sum / N
    train_mu = mu_sum / gamma_sum.unsqueeze(-1)
    train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

    print("N:", N)
    print("phi :", train_phi)
    print("mu : ", train_mu)
    print("covariance :", train_cov)

    # ============================ #
    # Get sample energy for test set
    # ============================ #
    test_energy = []

    test_X = data_dict['test']
    num_batches = test_X.shape[0] // batch_size + 1
    print('Size of test ', test_X.shape[0])
    for b in range(num_batches):
        input_data = test_X[b * batch_size: (b + 1) * batch_size]
        input_data = FT(input_data).to(_DEVICE)
        enc, dec, z, gamma = dagmm_obj(input_data)
        sample_energy, cov_diag = dagmm_obj.compute_energy(
            z,
            phi=train_phi,
            mu=train_mu,
            cov=train_cov,
            size_average=False
        )
        test_energy.append(sample_energy.data.cpu().numpy())

    test_energy = np.concatenate(test_energy, axis=0)
    print('test_energy', test_energy.shape)
    auc_list = []
    # ===========
    # Get per sample energy of the anomalies
    # ===========

    for idx in range(1, num_anom_sets + 1):
        key = 'anom_' + str(idx)
        anom_X = data_dict[key]
        anom_energy = []
        num_batches = anom_X.shape[0] // batch_size + 1
        for b in range(num_batches):
            input_data = anom_X[b * batch_size: (b + 1) * batch_size]
            input_data = FT(input_data).to(_DEVICE)
            enc, dec, z, gamma = dagmm_obj(input_data)
            sample_energy, cov_diag = dagmm_obj.compute_energy(
                z,
                phi=train_phi,
                mu=train_mu,
                cov=train_cov,
                size_average=False
            )
            anom_energy.append(sample_energy.data.cpu().numpy())

        anom_energy = np.concatenate(anom_energy, axis=0)
        auPR = eval.eval(anom_energy, test_energy, order='descending')
        auc_list.append(auPR)
        print("AUC : {:0.4f} ".format(auPR))

    _mean = np.mean(auc_list)
    _std = np.std(auc_list)
    print(' Mean AUC ', np.mean(auc_list))
    print(' AUC std', np.std(auc_list))
    return (_mean, _std)



def execute(DATA_SET, id, K, model_config, anom_perc, num_anomaly_sets ):
    global GPU_COUNT
    _DEVICE = torch.device("cuda:0")
    data_dict, _ = data_fetcher.get_data(
        DATA_SET,
        set_id=id,
        num_anom_sets=num_anomaly_sets,
        anomaly_perc=anom_perc
    )

    model_obj = get_model_obj(
        data_dict,
        model_config,
        K,
        _DEVICE
    )

    mean_aupr, std = test_eval(model_obj, data_dict, num_anomaly_sets, _DEVICE)
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
parser.add_argument(
    '--num_jobs',
    type=int,
    default=-1,
    help='Number of runs'
)

# =========================================

args = parser.parse_args()
DATA_SET = args.DATA_SET
num_runs = args.num_runs
num_jobs = args.num_jobs

if num_jobs == -1:
    num_jobs = num_runs

LOG_FILE = 'log_results_{}.txt'.format(DATA_SET)
LOGGER = logger_utils.get_logger(LOG_FILE,'DAGMM')

LOGGER.info(DATA_SET)
config_file = 'config.yaml'
with open(config_file, 'r') as fh:
    config = yaml.safe_load(fh)

num_anomaly_sets = config[DATA_SET]['num_anomaly_sets']
anomaly_ratio = config[DATA_SET]['anomaly_ratio']
model_config = config[DATA_SET]['dagmm']

anom_perc = 100 * anomaly_ratio/(1+anomaly_ratio)
step=1
K_values = np.arange(1,5+step,step)
K_vs_auc = []
for K in K_values:
    K = int(K)
    LOGGER.info('Setting K :: {}'.format(K))
    _res_ = Parallel(n_jobs=num_jobs)(delayed(execute)(
        DATA_SET, id, K, model_config, anom_perc, num_anomaly_sets ) for id in range(1,num_runs+1)
    )

    results = np.array(_res_)
    mean_all_runs = np.mean(results[:,0])
    _std = np.std(results[:, 0])
    LOGGER.info(' Runs {}: Mean: {:4f} | Std {:4f}'.format(num_runs, mean_all_runs, _std))
    print('Mean AuPR over {} runs {:4f}'.format(num_runs, mean_all_runs))
    print('Details: ', results[:,0])
    K_vs_auc.append((K, mean_all_runs))

K_vs_auc = np.array(K_vs_auc)
LOGGER.info('nu vs AuPR '+ str(K_vs_auc[:,0]) +  str(K_vs_auc[:,1]))
logger_utils.close_logger(LOGGER)
