import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')
from pandarallel.utils.inliner import inline
from tqdm import tqdm
from collections import OrderedDict
try:
    from data_fetcher import data_fetcher
except:
    from .data_fetcher import data_fetcher
try:
    from model import model_2_v1 as Model
except:
    from .model import model_2_v1 as Model

from pathlib import Path
import multiprocessing
from pprint import pprint
import torch
import math
import yaml

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)

def visualize(model_obj, x1, x2):

    from sklearn.decomposition import TruncatedSVD
    import matplotlib.pyplot as plt

    x1 = model_obj.get_compressed_embedding(x1)
    x2 = model_obj.get_compressed_embedding(x2)

    svd = TruncatedSVD(n_components=2, n_iter=100, random_state=42)
    x = np.vstack([x1, x2])
    x3 = svd.fit_transform(x)

    plt.figure(figsize=[10, 10])
    plt.scatter(x3[:len(x1), 0], x3[:len(x1), 1], c='g', alpha=0.95)
    plt.scatter(x3[len(x1):, 0], x3[len(x1):, 1], c='r', alpha=0.35)
    plt.show()

    svd = TruncatedSVD(n_components=3, n_iter=100, random_state=42)
    x = np.vstack([x1, x2])
    x3 = svd.fit_transform(x)
    fig = plt.figure(figsize=[14, 14])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x3[:len(x1), 0], x3[:len(x1), 1], x3[:len(x1), 2], c='g', alpha=0.95, marker='o')
    ax.scatter(x3[len(x1):, 0], x3[len(x1):, 1], x3[len(x1):, 2], c='r', alpha=0.35, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    return


def create_config(
        data_set
):
    # Should return :
    # data_dict
    # meta_data_df [column, dimension]

    config_file = 'architecture_config.yaml'
    
    with open(config_file, 'r') as fh :
        config = yaml.safe_load(fh)
    config = config[data_set]
    latent_dim = config['ae_latent_dimension']

    data_dict, meta_data_df = data_fetcher.get_data(data_set,True)

    # discrete_columns : { column_name : num_categories }
    discrete_dims = OrderedDict ({k: v for k, v in zip(list(meta_data_df['column']), list(meta_data_df['dimension']))})
    num_discrete_columns = len(discrete_dims)
    real_dims = len(data_dict['train'].columns) - sum(discrete_dims.values())

    # ---------------
    # encoder_structure_config['ip_layers']
    # Format :
    # [ 'emb|onehot', num_categories, [ embedding dimension ]
    # ---------------
    encoder_structure_config = {
        'real_dims': real_dims,
        'discrete_dims': discrete_dims,
        'encoder_FCN_to_latent': config['encoder_FCN_to_latent'],
        'ae_latent_dimension' :  config['ae_latent_dimension'],
        'encoder_discrete_xform' : config['encoder_discrete_xform'],
        'encoder_real_xform' : config['encoder_real_xform']
    }

    # ======================================================
    # Set decoder structure
    # =========

    decoder_structure_config = {
        'real_dims': real_dims,
        'discrete_dims': discrete_dims,
        'decoder_FC_from_latent': config['decoder_FC_from_latent'],
        'decoder_discrete_xform' : config['decoder_discrete_xform'],
        'decoder_real_xform' : config['decoder_real_xform'],
        'ae_latent_dimension': config['ae_latent_dimension']
    }

    # ================
    # Format decoder_field_layers:
    # { idx : [[dim1,dim2], op_activation ]
    # ================
    loss_structure_config = {
        'discrete_dims' : discrete_dims,
        'real_loss_func': config['real_loss_func'],
    }


    return encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim

encoder_structure_config, decoder_structure_config, loss_structure_config, latent_dim = create_config('kddcup')
pprint(encoder_structure_config)
pprint(decoder_structure_config)

# ======================================= #

data_dict, _ = data_fetcher.get_data('kddcup',True)
train_df = data_dict['train']
train_X = train_df.values


ae_model =  Model(
    DEVICE,
    latent_dim,
    encoder_structure_config,
    decoder_structure_config,
    loss_structure_config,
    optimizer='Adam',
    batch_size=64,
    num_epochs=10,
    learning_rate=0.1,
    l2_regularizer=0.01
)
print(ae_model)
exit(1)


losses = ae_model.train_model(
    train_X
)