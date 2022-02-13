#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datetime import datetime
import sys
import subprocess
import gc
import glob
import json
import math
import random
import time
from datetime import datetime
from collections import Counter, defaultdict
from typing import Union, Tuple, Optional, List

import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
get_ipython().magic(u'matplotlib inline')
sns.set(style="whitegrid")

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sys.path.append("../input/my-pipeline/code/")


# In[2]:


class Config:
    competition_name = "h-and-m-personalized-fashion-recommendations"
    exp_name = "exp1"
    input_dir = f"../input/h-and-m-personalized-fashion-recommendations/"
    my_input_dir = f"../input/create-dataset1-hmpf/"
    output_dir = f"./" 
    model_dir = f"./"
    log_dir = f"./"
    seed = 42
    DEBUG = False


# In[3]:


from utils.logger import Logger
from utils.reduce_mem_usage import reduce_mem_usage
from utils.timer import Timer
from utils.util import decorate, Util
logger = Logger(path=Config.log_dir, exp_name=Config.exp_name)


# In[4]:


with Timer(logger=logger, prefix="pip install implicit"):
    get_ipython().system('pip install --upgrade -q implicit')


# In[5]:


import implicit
from scipy.sparse import coo_matrix
from implicit.evaluation import mean_average_precision_at_k


# In[6]:


with Timer(logger=logger, prefix="loading pkl"):
    transaction_df = Util.load_df(Config.my_input_dir + "transaction.pkl", is_pickle=True)
    customer_df = Util.load_df(Config.my_input_dir + "customer.pkl", is_pickle=True)
    art_df = Util.load_df(Config.my_input_dir + "art.pkl", is_pickle=True)
    train_df = Util.load_df(Config.my_input_dir + "train.pkl", is_pickle=True)


# In[7]:


with Timer(logger, prefix="category ==> datetime"):
    transaction_df["t_dat"] = pd.to_datetime(transaction_df["t_dat"].astype("object"))
#     train_df["t_dat"] = pd.to_datetime(train_df["t_dat"].astype("object"))


# In[8]:


with Timer(logger, prefix="set user_id and item_id"):
    ALL_USER = customer_df["customer_id"].unique().tolist()
    ALL_ITEM = art_df["article_id"].unique().tolist()
    print(len(ALL_USER), len(ALL_ITEM))

    user_ids = dict(list(enumerate(ALL_USER)))
    item_ids = dict(list(enumerate(ALL_ITEM)))

    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    transaction_df["user_id"] = transaction_df["customer_id"].map(user_map)
    transaction_df["item_id"] = transaction_df["article_id"].map(item_map)


# In[9]:


def to_user_item_coo(df: pd.DataFrame):
    row = df["user_id"].values
    col = df["item_id"].values
    data = np.ones(df.shape[0])
    coo_train = coo_matrix((data, (row, col)), shape=(len(ALL_USER), len(ALL_ITEM)))
    return coo_train

def split_data(df: pd.DataFrame, validation_days: int = 7):
    validation_cut = df["t_dat"].max() - pd.Timedelta(validation_days)
    df_train = df[df["t_dat"] < validation_cut]
    df_valid = df[df["t_dat"] >= validation_cut]
    return df_train, df_valid

def get_valid_matrices(df: pd.DataFrame, validation_days: int = 7):
    df_train, df_valid = split_data(df, validation_days=validation_days)
    coo_train = to_user_item_coo(df_train)
    coo_valid = to_user_item_coo(df_valid)
    
    csr_train = coo_train.tocsr()
    csr_valid = coo_valid.tocsr()
    
    return {
        "coo_train": coo_train,
        "csr_train": csr_train,
        "csr_valid": csr_valid
    }

def validate(matrices, factors=20, iterations=20, regularization=0.01, show_progress=True, logger=None):
    coo_train, csr_train, csr_valid = matrices["coo_train"], matrices["csr_train"], matrices["csr_valid"]
    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                iterations=iterations,
                                                regularization=regularization,
                                                random_state=Config.seed)
    model.fit(coo_train, show_progress=show_progress)
    map12 = mean_average_precision_at_k(model, csr_train, csr_valid, K=12, show_progress=show_progress, num_threads=4)
    if logger is not None:
        logger.info(f"Factors: {factors:>3} - Iterations: {iterations:>2} - Regularization: {regularization:4.3f} ==> MAP@12: {map12:6.5f}")
    else:
        print(f"Factors: {factors:>3} - Iterations: {iterations:>2} - Regularization: {regularization:4.3f} ==> MAP@12: {map12:6.5f}")
    
    return map12


# In[10]:


def find_best_parameter(df: pd.DataFrame, logger=None, debug=False):
    matrices = get_valid_matrices(df=df)
    if debug:
#         map12 = validate(matrices=matrices, factors=40, iterations=3, regularization=0.01, show_progress=True)
        best_params = {"factors": 40, "iterations": 3, "regularization": 0.01}
        if logger is not None:
            logger.info(f"Best Map@12 found. Updating: {best_params}")
        else:
            print(f"Best Map@12 found. Updating: {best_params}")
    else:        
        for factors in [40, 50, 60, 100, 200, 500, 1000]:
            for iterations in [3, 12, 14, 15, 20]:
                for regulaization in [0.01]:
                    map12 = validate(matrices=matrices, factors=factors, iterations=iterations, regularization=regulaization, show_progress=False)
                    if map12 > best_map12:
                        best_map12 = map12
                        best_params = {"factors": factors, "iterations": iterations, "regularization": regulaization}
                        if logger is not None:
                            logger.info(f"Best Map@12 found. Updating: {best_params}")
                        else:
                            print(f"Best Map@12 found. Updating: {best_params}")
    del matrices
    gc.collect()
    return best_params


# In[11]:


def train(df: pd.DataFrame, best_params=None, show_progress=True):
    coo_train = to_user_item_coo(df)
    csr_train = coo_train.tocsr()
    model = implicit.als.AlternatingLeastSquares(**best_params, random_state=Config.seed)
    model.fit(coo_train, show_progress=show_progress)
    return model, csr_train


# In[12]:


def submit(model, csr_train, submission_name="submissions.csv"):
    preds = []
    batch_size = 2000
    to_generate = np.arange(len(ALL_USER))
    print(len(to_generate))
    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx : startidx + batch_size]
        ids, scores = model.recommend(batch, csr_train[batch], N=12, filter_already_liked_items=False)
        for i, userid in enumerate(batch):
            customer_id = user_ids[userid]
            user_items = ids[i]
            article_ids = [item_ids[item_id] for item_id in user_items]
            preds.append((customer_id, " ".join(article_ids)))
    df_preds = pd.DataFrame(preds,  columns=["customer_id", "prediction"])
    print(df_preds.shape)
    assert df_preds.shape[0] == 1371980
    df_preds.to_csv(submission_name, index=False)
    display(df_preds.head())
    return df_preds


# In[13]:


def main():
    with Timer(logger, prefix="find best parameters"):
        best_params = find_best_parameter(df=transaction_df, logger=logger, debug=True)
    with Timer(logger, prefix="train"):
        model, csr_train = train(transaction_df, best_params=best_params, show_progress=True)
    with Timer(logger, prefix="create submission.csv"):
        df_preds = submit(model, csr_train)


# In[14]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




