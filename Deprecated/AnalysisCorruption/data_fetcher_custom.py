import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
sys.path.append('./..')
sys.path.append('./../..')
sys.path.append('./../../..')
import yaml
from tqdm import tqdm
try:
    from model_6 import neg_sample_gen
except:
    from .model_6 import neg_sample_gen
from pandarallel import pandarallel
pandarallel.initialize()
from joblib import delayed,Parallel
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
try:
    from common_utils import utils
except:
    from .common_utils import utils
# =============================================== #

def get_data(
        data_set,
        anomaly_ratio = 0.1,
        num_anom_sets = 5,
        corruption_perc = 1,
        num_neg_samples = 10,
        exclude_corrupted = False
):

    save_dir = 'model_use_Data/{}/mixed_data_{}'.format(data_set, corruption_perc)
    path_obj = Path(save_dir)

    path_obj.mkdir(
        exist_ok=True, parents=True
    )

    df_dict = {}
    pos_file_path_N = os.path.join(save_dir, 'pos_samples_N.csv')
    neg_file_path_N = os.path.join(save_dir, 'neg_samples_N.csv')
    pos_files_dir_N = os.path.join(save_dir, 'pos_samples_N')
    neg_files_dir_N = os.path.join(save_dir, 'neg_samples_N')
    pos_file_path_C = os.path.join(save_dir, 'pos_samples_C.csv')
    neg_file_path_C = os.path.join(save_dir, 'neg_samples_C.csv')
    pos_files_dir_C = os.path.join(save_dir, 'pos_samples_C')
    neg_files_dir_C = os.path.join(save_dir, 'neg_samples_C')

    DATA_LOC = './../{}/processed_mixed_{}'.format(data_set, corruption_perc)

    if not os.path.exists(DATA_LOC):
        print('ERROR :', DATA_LOC)
        exit(1)

    meta_data = pd.read_csv(
        os.path.join(DATA_LOC, 'data_dimensions.csv'),
        index_col=None,
        low_memory=False
    )

    # ====================================
    # Set aside the anomaly data first
    # Anomaly data is one hot encoded
    # ====================================
    f_path = os.path.join(DATA_LOC, 'data_onehot.csv')
    data_df = utils.fetch_csv(f_path)
    anom_data = data_df.loc[data_df['label'] == 1]
    del anom_data['label']

    # =====================================
    # Non 1-hot encoded data
    # N: Normal
    # C: Corrupted
    # =====================================
    if (os.path.exists(pos_file_path_N) or os.path.exists(pos_files_dir_N))   and \
        (os.path.exists(neg_file_path_N) or os.path.exists(neg_files_dir_N) ) and \
        (os.path.exists(pos_file_path_C) or os.path.exists(pos_files_dir_C))  and \
        (os.path.exists(neg_file_path_C) or os.path.exists(neg_files_dir_C)) :

        df_pos_N = utils.fetch_csv(pos_file_path_N)
        df_neg_N = utils.fetch_csv(neg_file_path_N)
        print(len(df_pos_N), len(df_neg_N))
        num_data_N = len(df_pos_N)
        pos_N = df_pos_N.values
        neg_N = df_neg_N.values
        neg_N = np.reshape(neg_N, [num_data_N, num_neg_samples, pos_N.shape[1]])

        df_pos_C = utils.fetch_csv(pos_file_path_C)
        df_neg_C = utils.fetch_csv(neg_file_path_C)
        num_data_C = len(df_pos_C)
        pos_C = df_pos_C.values
        neg_C = df_neg_C.values
        neg_C = np.reshape(neg_C, [num_data_C, num_neg_samples, pos_C.shape[1]])

    else:
        f_path = os.path.join(DATA_LOC, 'data.csv')
        data_df = utils.fetch_csv(f_path)
        from collections import Counter
        print(Counter(data_df['label']))
        N_data = data_df.loc[data_df['label'] == 0]
        C_data = data_df.loc[data_df['label'] == 2]
        print(len(N_data), len(C_data))

        try:
            del N_data['label']
            del C_data['label']
        except:
            pass

        discrete_dims = OrderedDict(
            {k: v for k, v in zip(
                list(meta_data['column']), list(meta_data['dimension'])
            )}
        )

        pos_C, neg_C = neg_sample_gen.generate_pos_neg_data(
            C_data,
            discrete_dims,
            num_samples=num_neg_samples
        )
        print(' >>>', pos_C.shape, neg_C.shape)
        pos_N, neg_N = neg_sample_gen.generate_pos_neg_data(
            N_data,
            discrete_dims,
            num_samples=num_neg_samples
        )



        neg_N_np = np.array(neg_N)
        num_data_N = pos_N.shape[0]
        neg_N_np = np.reshape(neg_N_np, [num_data_N * num_neg_samples, -1])
        df_neg_N = pd.DataFrame(data=neg_N_np)
        df_pos_N = pd.DataFrame(data=pos_N)
        utils.save_csv(df_pos_N, pos_file_path_N)
        utils.save_csv(df_neg_N, neg_file_path_N)

        neg_C_np = np.array(neg_C)
        num_data_C = pos_C.shape[0]
        neg_C_np = np.reshape(neg_C_np, [num_data_C * num_neg_samples, -1])

        df_neg_C = pd.DataFrame(data=neg_C_np)
        df_pos_C = pd.DataFrame(data=pos_C)
        utils.save_csv(df_pos_C, pos_file_path_C)
        utils.save_csv(df_neg_C, neg_file_path_C)

    # -----
    # Recreate the train and test sets
    idx = np.arange(num_data_N)
    np.random.shuffle(idx)

    train_idx = idx[:int(0.7 * idx.shape[0])]
    test_idx = idx[int(0.7 * idx.shape[0]):]

    train_arr_pos = pos_N[train_idx]
    train_arr_neg = neg_N[train_idx]

    # Add in corruption to training data
    if not exclude_corrupted :
        pos_X = np.vstack([train_arr_pos, pos_C])
        neg_X = np.vstack([train_arr_neg, neg_C])
    else:
        pos_X = train_arr_pos
        neg_X = train_arr_neg
        
    test_X = pos_N[test_idx]

    df_dict['test'] = test_X
    df_dict['train'] = pos_X

    anom_size = int(anomaly_ratio * test_X.shape[0])
    for i in range(1, num_anom_sets + 1):
        _df = anom_data.sample(n=anom_size)
        df_dict['anom_' + str(i)] = _df

    return pos_X, neg_X, df_dict, meta_data


# pos_X, neg_X, df_dict, meta_data = get_data(
#         data_set='kddcup',
#         anomaly_ratio = 0.2,
#         num_anom_sets = 5,
#         corruption_perc = 1,
#         num_neg_samples = 10
# )
# print(pos_X.shape, neg_X.shape)