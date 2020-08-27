import pandas as pd
import numpy as np
import os 
import sys
import multiprocessing
import yaml
import shutil

def save_csv(df, file_path):
    df.to_csv(file_path, index=False)
    _size = os.stat(file_path).st_size / (1024*1024)
    print('Size {:.3f}', _size, ' MB ')
    max_size = 99
    
    if _size > max_size :
        os.remove(file_path)
        # Create a directory 
        dir_path = file_path.replace('.csv','')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path) 
        os.mkdir(dir_path)
        num_chunks = _size //max_size + 1
        chunk_size = int(len(df)//num_chunks)
        
        print( '> ', df.shape[0], chunk_size )
        # chunk the df
        idx = 0 
        for start in range(0, df.shape[0], chunk_size):
            df_subset = df.iloc[start:start + chunk_size]
            df_subset.to_csv( os.path.join(dir_path, 'chunk_{}.csv'.format(idx)), index=False)
            idx += 1
    return _size

def fetch_csv(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=False)
        return df 
    
    dir_path = file_path.replace('.csv','')
    # chunk the df
    df = None
    idx = 0 
    while True:
        try:
            df_subset = pd.read_csv( 
                os.path.join(dir_path, 'chunk_{}.csv'.format(idx)), 
                index_col=None)
           
            if df is None:
                df = df_subset
            else:
                df = df.append(df_subset, ignore_index=True)
            idx += 1
        except:
            break
            
    return df