# ==========================
# This is to run the base model
# ==========================

import yaml

try:
    from ..data_fetcher import data_fetcher
except:
    import data_fetcher
from pprint import pprint
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print('Current device  >> ', DEVICE)
try:
    from .deprecated import base_DAGMM
except:
    import base_DAGMM


# ============================
#  create config for DAGMM_base
# ============================


def create_config(
        data_set
):
    # Should return :
    # data_dict
    # meta_data_df [column, dimension]

    config_file = 'architecture_config.yaml'

    with open(config_file, 'r') as fh:
        config = yaml.safe_load(fh)

    data_dict, meta_data_df = data_fetcher.get_data(data_set)

    # discrete_columns : { column_name : num_categories }
    discrete_column_dims = {
        k: v for k, v in
        zip(list(meta_data_df['column']), list(meta_data_df['dimension']))
    }

    num_discrete_columns = len(discrete_column_dims)
    num_real_columns = len(data_dict['train'].columns) - num_discrete_columns
    latent_dim = config[data_set]['ae_latent_dimension']

    encoder_structure_config = {}
    encoder_structure_config['discrete_column_dims'] = discrete_column_dims
    encoder_structure_config['num_discrete'] = num_discrete_columns
    encoder_structure_config['num_real'] = num_real_columns
    encoder_structure_config['encoder_layers'] = {
        'activation': config[data_set]['encoder_layers']['activation'],
        'layer_dims': config[data_set]['encoder_layers']['layer_dims'] + [latent_dim]
    }

    # ======================================================
    # Set decoder structure
    # =========

    decoder_structure_config = {}
    final_op_dims = num_real_columns + sum(discrete_column_dims.values())
    decoder_structure_config['discrete_column_dims'] = discrete_column_dims
    decoder_structure_config['num_discrete'] = num_discrete_columns
    decoder_structure_config['num_real'] = num_real_columns
    decoder_structure_config['decoder_layers'] = {
        'activation': config[data_set]['decoder_layers']['activation'],
        'layer_dims': [latent_dim] + config[data_set]['decoder_layers']['layer_dims'] + [final_op_dims]
    }
    decoder_structure_config['final_output_dim'] = final_op_dims

    # =====================
    # GMM
    # =====================
    gmm_input_dims = latent_dim + 2
    activation = config[data_set]['gmm']['FC_layer']['activation']
    num_components = config[data_set]['gmm']['num_components']
    FC_layer_dims = [gmm_input_dims] + config[data_set]['gmm']['FC_layer']['dims'] + [num_components]
    FC_dropout = config[data_set]['gmm']['FC_dropout']
    gmm_structure_config = {
        'num_components' : num_components,
        'FC_layer_dims' : FC_layer_dims,
        'FC_dropout' : FC_dropout,
        'FC_activation' : activation

    }
    loss_structure_config = []

    for column, dim in discrete_column_dims.items():
        loss_structure_config.append(
            {
                'dim': dim,
                'type': 'onehot'
            }
        )
    loss_structure_config.append(
        {
            'dim': num_real_columns,
            'type': 'real'
        }
    )

    return encoder_structure_config, decoder_structure_config, gmm_structure_config, loss_structure_config, latent_dim

# =================================== #
data_set = 'kddcup'
data, _ = data_fetcher.get_data(data_set)
train_df = data['train']
train_X = train_df.values

encoder_structure_config, decoder_structure_config, gmm_structure_config, loss_structure_config,latent_dim = create_config(
    data_set
)
pprint(encoder_structure_config)
pprint(decoder_structure_config)
# =================================== #

ae_model = base_DAGMM.DAGMM_base_model(
    DEVICE,
    latent_dim,
    encoder_structure_config,
    decoder_structure_config,
    gmm_structure_config,
    loss_structure_config,
    optimizer='Adam',
    batch_size=1024,
    num_epochs=10,
    learning_rate=0.05,
)

losses = ae_model.train_model(
    train_X
)

# =======================
# Save model
# =======================
model_save_path = 'data_set_dagmm.pkl'
torch.save(ae_model, model_save_path, pickle_protocol=4)
# =======================
ae_model = torch.load(model_save_path)
