import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel,delayed
import multiprocessing
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
import sys
sys.path.append('./..')
sys.path.append('./../..')
import warnings

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

def aux_gen(
    row,
    discrete_dim_list, # Ordered
    num_real_dims,
    column_encoder,
    num_samples = 10
):
    row_vals = row.values
    num_cat = len(discrete_dim_list)
    real_part = row_vals[-num_real_dims:]
    cat_part = row_vals[:num_cat]
    nr = num_real_dims//2
    ns = num_samples

    # ======
    a = num_real_dims//4
    b = num_real_dims//4
    c = num_real_dims - (a + b)
    # Adding -.5 to shift noise to be between -.5 to .5
    noise = np.concatenate(
        [np.random.random_sample([ns,a])  + -0.5, 
         np.random.random_sample([ns,b])  +  0.5, 
         np.zeros([ns,c])],
        axis=1
    )
   
    for i in range(ns):
        np.random.shuffle(noise[i])
    # ---
    # noise shape [ ns, num_real_dims ]

    part_r_duplicated = np.tile(real_part, ns).reshape([ns, num_real_dims])
    part_r_duplicated = part_r_duplicated + noise
    
    P = [ np.power( _/sum(discrete_dim_list), 0.75)  for _ in discrete_dim_list]  
    P = [ _/sum(P) for _ in P]  
    part_c_duplicated = np.tile(cat_part,ns).reshape([ns,num_cat])
   
    # ------------------------------
    # For categorical variables
    # ------------------------------
    res = []
    for i in range(ns):
        _copy = np.array(row_vals)[:num_cat]
        if num_cat < 3 :
            pert_idx = np.random.choice( list(np.arange(num_cat)) , size=1, replace = False, p = P)
        else:
            pert_idx = np.random.choice(
                list(np.arange(num_cat)),
                np.random.randint(1,num_cat//2+1),
                replace=False,
                p = P
            )

        for j in pert_idx:
            _copy[j] = np.random.choice(
                np.arange(discrete_dim_list[j]),1
            )
        part_c_duplicated[i] = _copy
        
        
    _samples = np.concatenate([part_c_duplicated, part_r_duplicated] ,axis=1)
    row_vals = np.reshape( row.values,[1,-1] )

    samples = np.concatenate([ row_vals, _samples ],axis=0)
    sample_cat_part = samples[:, : num_cat]
    samples_real_part = samples[:, -num_real_dims: ]

    # =========================
    # Do a 1-hot transformation
    # Drop binary columns
    # =========================

    onehot_xformed = column_encoder.fit_transform(sample_cat_part)
    print('>>> 1-0 part ', onehot_xformed.shape)
    print('>>> Real part ', samples_real_part[:3])
    samples = np.concatenate([onehot_xformed, samples_real_part],axis=1)
    
    pos = samples[0]
    neg = samples[1:]
    return pos, neg

def generate_pos_neg_data (
        train_df,
        cat_domain_dims,
        num_samples=10
):
    try:
        del train_df['label']
    except:
        pass
    
 
    num_cat = len(cat_domain_dims)
    num_real = len(train_df.columns) - num_cat
    
    oh_encoder_list = []
    idx = 0
    for _ , dim in cat_domain_dims.items():
        if dim ==2 :
            _drop = 'first'
        else:
            _drop = None
        name = "oh_"+str(idx) 
        oh_encoder = OneHotEncoder(
            np.reshape( list(range(dim)),[1,-1] ),
            sparse=False,
            drop=_drop
        ) 
        oh_encoder_list.append((name, oh_encoder, [idx]))
        idx +=1
    column_encoder = ColumnTransformer(
        oh_encoder_list
    )
                                
    discrete_dim_list = list(cat_domain_dims.values())
    n_jobs = multiprocessing.cpu_count()
    
    res = Parallel(n_jobs)(delayed(aux_gen)(
            row, discrete_dim_list, num_real, column_encoder, num_samples
        ) for i,row in tqdm(train_df.iterrows(), total=train_df.shape[0])
    )

#     res = []
#     for i,row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
#         r = aux_gen(
#             row, discrete_dim_list, num_real, column_encoder , num_samples
#         ) 
#         res.append(r)
        
    pos = []
    neg = []
    for r in res:
        pos.append(r[0])
        neg.append(r[1])

    pos = np.array(pos)
    neg = np.array(neg)
    # print(pos.shape)
    # print(neg.shape)
    return pos, neg

# cat = [ list(range(_)) for _ in [10,15,10]]
#
# oh_encoder = OneHotEncoder(
#         cat,
#         sparse=False
#     )
# arr = np.array([[7,4,5],[4,5,6],[2,9,8]])
# onehot_xfromed = oh_encoder.fit_transform(arr)
# print(onehot_xfromed.shape)
