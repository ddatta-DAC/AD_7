import pandas as pd
import os
import numpy as np
import sys
sys.path.append('./../..')
sys.path.append('./..')
from tqdm import tqdm
import multiprocessing
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from collections import Counter
from pandarallel import pandarallel
pandarallel.initialize()
from scipy import sparse

def get_data(
        data_set,
        set_id=1,
        anomaly_perc=10,
        num_anom_sets=5,
        data_sparse=False
):
    DATA_LOC = './../{}/processed_sets/set_{}'.format(data_set,str(set_id))
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


# df_dict,meta_data = get_data('kddcup',True)
#
# print(len(df_dict['anom_1'].columns))
# print(len(df_dict['train'].columns))
