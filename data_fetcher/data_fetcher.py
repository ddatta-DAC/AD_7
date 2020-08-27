import pandas as pd
import numpy as np
import os
import sys
sys.path.append('./..')
sys.path.append('./../..')
from tqdm import tqdm
from pathlib import Path
import multiprocessing
from pandarallel import pandarallel
pandarallel.initialize()
from sklearn.preprocessing import StandardScaler
from RP_1.common_utils import utils
from sklearn.model_selection import train_test_split
try:
    from common_utils import utils
except:
    from .common_utils import utils
# ========================================= #


# -------------
# Modify Port for CCIDS data
# ------------
def set_port(row, col=None):
    if type(row[col]) == str:
        print(row[col])
    if row[col] <= 1023:
        return row[col]
    elif row[col] > 1023 and row[col] <= 49151:
        return 1024
    elif row[col] > 49151:
        return 1025
    return -1


def get_data_v1():
    standardize_data = True

    DATA_PATH = '../CICIDS/model_use_data/exp1'

    train_df = pd.read_csv(os.path.join(
        DATA_PATH,
        'train_data.csv'
    ),
        index_col=None
    )
    test_norm_df = pd.read_csv(os.path.join(
        DATA_PATH,
        'test_data_normal.csv'
    ),
        index_col=None
    )
    test_anom_df = pd.read_csv(os.path.join(
        DATA_PATH,
        'test_data_anomalies.csv'
    ),
        index_col=None
    )

    df_dict = {
        'train': train_df,
        'test_norm': test_norm_df,
        'test_anom': test_anom_df
    }

    columns = list(train_df.columns)
    non_numeric_cols = [
        'Flow_ID',
        'Source_IP',
        'Source_Port',
        'Destination_IP',
        'Destination_Port', 'Protocol', 'Timestamp', 'Label'
    ]
    numerical_cols = [_ for _ in columns if _ not in non_numeric_cols]

    # ===============
    # Order of the fields is important
    # ===============

    discrete_columns = ['Source_Port', 'Source_IP', 'Destination_IP', 'Destination_Port', 'Protocol']
    ordered_columns = discrete_columns + numerical_cols
    port_columns = ['Source_Port', 'Destination_Port']

    processed_df_dict = {}
    for key, _df in df_dict.items():
        df = _df.copy()
        df = df[ordered_columns]
        for port in port_columns:
            df[port] = df.parallel_apply(
                set_port,
                axis=1,
                args=(port,)
            )
        processed_df_dict[key] = df

    # Calculate the domain dimensions
    discrete_domain_sizes = {}
    val2id_dict = {}
    for dc in discrete_columns:
        _entity_set = set()
        for key, _df in processed_df_dict.items():
            _entity_set = _entity_set.union(set(_df[dc]))
        discrete_domain_sizes[dc] = len(_entity_set)
        # Convert to ids
        val2id_dict[dc] = {}
        for e in enumerate(_entity_set, 0):
            val2id_dict[dc][e[1]] = e[0]

    def replace_with_ids(value, ref_dict):
        return ref_dict[value]

    for key, df in processed_df_dict.items():

        for dc in discrete_columns:
            df[dc] = df[dc].parallel_apply(
                replace_with_ids,
                args=(val2id_dict[dc],)
            )

    # ============================================
    # Standardize the continuous values
    # ============================================

    columns_to_standardize = numerical_cols
    if standardize_data:
        for i in tqdm(range(len(columns_to_standardize))):
            col = columns_to_standardize[i]
            _df = processed_df_dict['train']
            _df = _df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

            values = list(_df[col])
            values = np.reshape(values, [-1, 1])
            scaler_obj = StandardScaler()
            scaler_obj.fit(values)

            for key, _df in processed_df_dict.items():
                df = _df.copy()
                df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
                values = list(df[col])
                values = np.reshape(values, [-1, 1])
                df[col] = np.reshape((scaler_obj.transform(values)), -1)
                processed_df_dict[key] = df

    return {'data_dict': processed_df_dict,
            'discrete_columns': discrete_domain_sizes,
            'real_columns': numerical_cols
            }


def get_data(
        data_set,
        one_hot=False,
        anomaly_ratio=0.1,
        num_anom_sets=5
):
    DATA_LOC = './../{}/processed'.format(data_set)
    if not os.path.exists(DATA_LOC):
        print('ERROR :', DATA_LOC)
        exit(1)
    if one_hot:
        f_path = os.path.join(DATA_LOC, 'data_onehot.csv')
    else:
        f_path = os.path.join(DATA_LOC, 'data.csv')

    data_df = utils.fetch_csv(f_path)
    normal_data = data_df.loc[data_df['label'] == 0]
    anom_data = data_df.loc[data_df['label'] == 1]
    del anom_data['label']
    del normal_data['label']

    train_df, test_df = train_test_split(normal_data, test_size=0.3)
    anom_size = int(anomaly_ratio * len(test_df))

    df_dict = {
        'train': train_df,
        'test': test_df,
    }
    for i in range(1,num_anom_sets+1):
        _df = anom_data.sample(n=anom_size)
        df_dict['anom_' + str(i)] = _df

    # Read in data characteristics
    # File has 2 columns:
    # column, dimension
    meta_data = pd.read_csv(
        os.path.join(DATA_LOC, 'data_dimensions.csv'),
        index_col=None,
        low_memory=False
    )

    return df_dict, meta_data


# df_dict,meta_data = get_data('kddcup',True)
#
# print(len(df_dict['anom_1'].columns))
# print(len(df_dict['train'].columns))
