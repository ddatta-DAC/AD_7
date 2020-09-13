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
import os
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize()
from scipy import sparse
from sklearn.metrics import auc
import matplotlib.pyplot as plt

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
    data_dict, meta_data_df = get_data(
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

# ===================
# Fetch data without negative samples
# ===================
def get_data(
        data_set,
        set_id=1,
        anomaly_perc=10,
        num_anom_sets=5,
        data_sparse=False
):
    DATA_LOC = './{}/processed_sets/set_{}'.format(data_set,str(set_id))
    if not os.path.exists(DATA_LOC):
        print('ERROR :', DATA_LOC)
        exit(1)
    else:
        print(' {} > '.format(data_set))

    train_data = sparse.load_npz(os.path.join(DATA_LOC,'train.npz'))
    test_data = sparse.load_npz(os.path.join(DATA_LOC, 'test.npz'))
    anom_data = sparse.load_npz(os.path.join(DATA_LOC, 'anom.npz'))
    if not data_sparse:
        train_data = train_data.todense()
        test_data = test_data.todense()
        anom_data = anom_data.todense()

    anom_size = int(anomaly_perc/(100-anomaly_perc) *  test_data.shape[0])
    data_dict = {
        'train': train_data,
        'test': test_data,
    }
    print(test_data.shape)
    for i in range(1,num_anom_sets+1):
        idx = np.arange(test_data.shape[0])
        np.random.shuffle(idx)
        idx = idx[:anom_size]
        data_dict['anom_' + str(i)] = anom_data[idx]

    # Read in data characteristics
    # File has 2 columns:
    # column, dimension
    meta_data = pd.read_csv(
        os.path.join(DATA_LOC, '../data_dimensions.csv'),
        index_col=None,
        low_memory=False
    )

    return data_dict, meta_data


def _normalize_(val, _min, _max):
    return (val - _min) / (_max - _min)


def evaluate(ae_model, data_dict, num_anomaly_sets,show_figure=False):
    auc_list = []
    test_norm_X = data_dict['test']

    for idx in range(1, num_anomaly_sets + 1):
        key = 'anom_' + str(idx)
        test_anom_df = data_dict[key]
        test_anom_X = test_anom_df
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
        if show_figure:
            try:
                plt.figure(figsize=[8, 6])
                plt.plot(R, P)
                plt.title('Precision Recall Curve  || auPR :' + "{:0.4f}".format(pr_auc), fontsize=15)
                plt.xlabel('Recall', fontsize=15)
                plt.ylabel('Precision', fontsize=15)
                plt.savefig('auPR.png')
                plt.show()
                plt.close()
            except:
                pass
        print("AUC : {:0.4f} ".format(pr_auc))
        auc_list.append(pr_auc)
    return auc_list