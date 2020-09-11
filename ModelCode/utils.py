import numpy as np
import sys
import os
sys.path.append('./..')
sys.path.append('./../..')
from collections import OrderedDict
try:
    from . import data_fetcher
except:
    import data_fetcher
import yaml
import matplotlib.pyplot  as plt
from sklearn.metrics import auc
import logging
import logging.handlers
from time import time
from datetime import datetime



# ==================================
# Obtain logger
# ==================================
def get_logger(LOG_FILE):

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    OP_DIR = os.path.join('Logs')

    if not os.path.exists(OP_DIR):
        os.mkdir(OP_DIR)

    handler = logging.FileHandler(os.path.join(OP_DIR, LOG_FILE))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Log start :: ' + str(datetime.now()))
    return logger

# ==================================
# To help timestamp
# ==================================
def log_time(logger):
    logger.info(str(datetime.now()) + '| Time stamp ' + str(time()))

# ==================================
# Close logger
# ==================================
def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    logging.shutdown()
    return

# ==================================
# Create configuration dict
# ==================================
def create_config(
        data_set
):
    # Should return :
    # data_dict
    # meta_data_df [column, dimension]
    config_file = 'architecture_config.yaml'

    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)
    config = config[data_set]
    latent_dim = config['ae_latent_dimension']
    data_dict, meta_data_df = data_fetcher.get_data(
        data_set,
        1
    )

    # discrete_columns : { column_name : num_categories }
    discrete_dims = OrderedDict({k: v for k, v in zip(list(meta_data_df['column']), list(meta_data_df['dimension']))})
    num_discrete_columns = len(discrete_dims)
    count_discrete_dims = 0
    for val in discrete_dims.values():
        if val == 2:
            count_discrete_dims += 1
        else:
            count_discrete_dims += val

    real_dims = data_dict['train'].shape[1] - count_discrete_dims

    # ---------------
    # encoder_structure_config['ip_layers']
    # Format :
    # [ 'emb|onehot', num_categories, [ embedding dimension ]
    # ---------------
    encoder_structure_config = {
        'real_dims': real_dims,
        'discrete_dims': discrete_dims,
        'encoder_FCN_to_latent': config['encoder_FCN_to_latent'],
        'ae_latent_dimension': config['ae_latent_dimension'],
        'encoder_discrete_xform': config['encoder_discrete_xform'],
        'encoder_real_xform': config['encoder_real_xform']
    }

    # ======================================================
    # Set decoder structure
    # =========

    decoder_structure_config = {
        'real_dims': real_dims,
        'discrete_dims': discrete_dims,
        'decoder_FC_from_latent': config['decoder_FC_from_latent'],
        'decoder_discrete_xform': config['decoder_discrete_xform'],
        'decoder_real_xform': config['decoder_real_xform'],
        'ae_latent_dimension': config['ae_latent_dimension']
    }

    # ================
    # Format decoder_field_layers:
    # { idx : [[dim1,dim2], op_activation ]
    # ================
    loss_structure_config = {
        'discrete_dims': discrete_dims,
        'real_loss_func': config['real_loss_func'],
        'real_dims': real_dims
    }

    return encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim
