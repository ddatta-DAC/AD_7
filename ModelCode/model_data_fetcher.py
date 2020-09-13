import os
import pandas as pd
from pathlib import Path
import multiprocessing
from pprint import pprint
import sys
sys.path.append('./..')
sys.path.append('./../..')

try:
    from data_fetcher import data_fetcher
except:
    from RP_1.data_fetcher import data_fetcher
try:
    from common_utils import utils
except:
    from RP_1.common_utils import utils
try:
    from . import neg_sample_gen
except:
    import neg_sample_gen

from pandarallel import pandarallel
pandarallel.initialize()
from joblib import delayed,Parallel
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from scipy import sparse

def fetch_model_data(
        data_set,
        set_id =1,
        num_neg_samples=10,
        anomaly_ratio = 0.2,
        num_anom_sets = 5
):
    # Save files
    
    save_dir = './' + data_set
    path_obj = Path(save_dir)
    path_obj.mkdir(
        exist_ok=True, parents=True
    )
    LOC = os.path.join('./{}/processed_sets/set_{}'.format(data_set, set_id))
    train_data = sparse.load_npz(os.path.join(LOC, 'train.npz'))
    test_data = sparse.load_npz(os.path.join(LOC, 'test.npz'))
    anom_data = sparse.load_npz(os.path.join(LOC, 'anom.npz'))
    train_data = train_data.todense()
    test_data = np.array(test_data.todense())
    anom_data = np.array(anom_data.todense())
    pos_x = train_data
    # ------------------
    # If negative samples do not exist , generate them
    # ------------------

    neg_samples_fname = 'neg_samples.npz'
    neg_file_path = os.path.join(LOC ,neg_samples_fname)

    d_df = pd.read_csv(
        os.path.join('./{}/processed_sets/'.format(data_set),'data_dimensions.csv'),
        index_col=None
    )
    meta_data = d_df
    data_dict = {}
    data_dict['train'] = train_data
    data_dict['test'] = test_data

    anom_size = int(test_data.shape[0]*anomaly_ratio)
    for i in range(1, num_anom_sets + 1):
        idx = np.arange(anom_data.shape[0])
        np.random.shuffle(idx)
        idx = idx[:anom_size]
        data_dict['anom_' + str(i)] = anom_data[idx]

    data_len = train_data.shape[0]
    if not os.path.exists(neg_file_path):
        cat_domain_dims = OrderedDict(
            {k: v for k, v in zip(
                list(d_df['column']), list(d_df['dimension']))
             }
        )
        # train_df is not oh encoded 
        train_df = pd.read_csv(os.path.join(LOC ,'train_data.csv'))               
        pos, neg = neg_sample_gen.generate_pos_neg_data(
                train_df,
                cat_domain_dims,
                num_samples=num_neg_samples
        )
        neg_reshaped =  sparse.csr_matrix(np.reshape(neg, [data_len * num_neg_samples,-1]))
        sparse.save_npz(neg_file_path, neg_reshaped)
        neg_x = neg
    else:
        neg = sparse.load_npz(neg_file_path)
        neg = np.array(neg.todense())
        print(neg.shape)
        neg_x = np.reshape(neg, [data_len, num_neg_samples, -1])

    data_dict['neg'] = neg_x
    return pos_x, neg_x, data_dict



 


