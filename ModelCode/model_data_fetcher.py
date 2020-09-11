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
        anomaly_ratio = 0.1,
        num_anom_sets = 5
):
    # Save files
    
    save_dir = './' + data_set
    path_obj = Path(save_dir)
    path_obj.mkdir(
        exist_ok=True, parents=True
    )
    
    # ------------------
    # If negative samples do not exist , generate them
    # ------------------
    
    LOC = os.path.join('./{}/processed_sets/set_{}'.format(data_set,set_id)
    neg_samples_fname = 'neg_samples.npz'
    neg_file_path = os.path.join(LOC ,neg_samples_fname)
    
  
    d_df = pd.read_csv(
        os.path.join('./{}/processed_sets/'.format(data_set),'data_dimensions.csv'),
        index=None
    )
    cat_domains = {k:v for k,v in zip(list(d_df['column']),list(d_df['dimension']))
                    
                
    if not os.path.exists(neg_file_path):
        # train_df is not oh encoded 
        train_df = pd.read_csv(os.path.join(LOC ,'train_data.csv'))               
        pos, neg = neg_sample_gen.generate_pos_neg_data(
                train_df,
                cat_domain_dims,
                num_samples=num_neg_samples
        )
        neg = sparse.csr_matrix()
    else:
        
        
    df_dict, meta_data = data_fetcher.get_data (
                data_set,
                one_hot=False,
                anomaly_ratio = anomaly_ratio,
                num_anom_sets = num_anom_sets
        )
    
    _df_dict_, _ = data_fetcher.get_data (
                data_set,
                one_hot=True,
                anomaly_ratio = anomaly_ratio,
                num_anom_sets = num_anom_sets
    )
    
    for i in range(num_anom_sets):
        key = 'anom_' + str(i+1)
        df_dict[key] = _df_dict_[key]
        
    train_df =  df_dict['train']
    test_df =  df_dict['test']
    all_data = train_df.append(test_df,ignore_index=True)
    
    if (os.path.exists(pos_file_path) or os.path.exists(pos_files_dir)) and (os.path.exists(neg_file_path) or os.path.exists(neg_files_dir) ):
        
        df_pos = utils.fetch_csv(pos_file_path)
        df_neg = utils.fetch_csv(neg_file_path)
        print(len(df_pos),len(df_neg))
        pos = df_pos.values
        neg = df_neg.values
        neg_x = np.reshape(neg, [pos.shape[0], num_neg_samples, pos.shape[1]])
    
    else:     

        # =================== #
    
        discrete_dims = OrderedDict (
            {k: v for k, v in zip(
                list(meta_data['column']), list(meta_data['dimension'])
        )}
        )

        pos, neg = neg_sample_gen.generate_pos_neg_data (
            all_data,
            discrete_dims,
            num_samples=num_neg_samples
        )

        neg_x = np.array(neg)
        num_data = pos.shape[0]
        neg = np.reshape(neg,[num_data*num_neg_samples, -1])
        df_neg = pd.DataFrame(data=neg)
        df_pos = pd.DataFrame(data=pos)
        utils.save_csv(df_pos, pos_file_path)
        utils.save_csv(df_neg, neg_file_path)

    num_data = pos.shape[0]
    # Recreate the train and test sets
    idx = np.arange(num_data)
    np.random.shuffle(idx)
    
    train_idx = idx[:int(0.7*idx.shape[0])]
    test_idx = idx[int(0.7*idx.shape[0]):]
    
    df_dict['train'] = pos[train_idx]
    df_dict['test'] = pos[test_idx]
    pos_x = pos[train_idx]
    neg_x = neg_x[train_idx]
 
    return pos_x, neg_x, df_dict

