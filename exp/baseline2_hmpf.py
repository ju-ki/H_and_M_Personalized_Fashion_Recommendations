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
import torch
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
    exp_name = "setup_cv"
    input_dir = f"../input/h-and-m-personalized-fashion-recommendations/"
    my_input_dir = f"../input/create-dataset1-hmpf/"
    output_dir = f"./" 
    model_dir = f"./"
    log_dir = f"./"
    seed = 42
    DEBUG = False


# In[3]:


def seed_everything(SEED):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
seed_everything(Config.seed)


# In[4]:


from utils.logger import Logger
from utils.reduce_mem_usage import reduce_mem_usage
from utils.timer import Timer
from utils.util import decorate, Util
logger = Logger(path=Config.log_dir, exp_name=Config.exp_name)


# In[5]:


with Timer(logger=logger, prefix="loading pkl"):
#     transaction_df = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv", parse_dates=["t_dat"], dtype={"article_id": str})
    transaction_df = Util.load_df(Config.my_input_dir + "transaction.pkl", is_pickle=True)
#     customer_df = Util.load_df(Config.my_input_dir + "customer.pkl", is_pickle=True)
#     art_df = Util.load_df(Config.my_input_dir + "art.pkl", is_pickle=True)
#     train_df = Util.load_df(Config.my_input_dir + "train.pkl", is_pickle=True)
transaction_df["t_dat"] = pd.to_datetime(transaction_df["t_dat"].astype(object))
transaction_df["article_id"] = transaction_df["article_id"].astype(str)
transaction_df["customer_id"] = transaction_df["customer_id"].astype(object)


# In[6]:


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# In[7]:


train = transaction_df[transaction_df["t_dat"] >= datetime(2020, 9, 1)]
print(train.shape)


# In[8]:


val_start_date = '2020-09-16'

valid = train.query(f"t_dat >= '{val_start_date}'").reset_index(drop=True) 
train = train.query(f"t_dat < '{val_start_date}'").reset_index(drop=True)

print(train.shape, valid.shape)


# In[9]:


positive_items_per_user = train.groupby(["customer_id"])["article_id"].apply(list)
positive_items_val = valid.groupby(["customer_id"])["article_id"].apply(list)


# In[10]:


from tqdm.notebook import tqdm
val_users = positive_items_val.keys()
val_items = []

for i, user in tqdm(enumerate(val_users)):
    val_items.append(positive_items_val[user])
print("Total users in validation", len(val_users))


# In[11]:


train["pop_factor"] = train["t_dat"].apply(lambda x: 1 / (datetime(2020, 9 ,16) - x).days)
pop_items_group = train.groupby(["article_id"])["pop_factor"].sum()
_, pop_items = zip(*sorted(zip(pop_items_group, pop_items_group.keys()))[::-1])


# In[12]:


from collections import Counter
outputs = []
cnt = 0

#9月限定で見たときのレコメンド
for i, user in tqdm(enumerate(val_users)):
    if user in positive_items_per_user.keys():
        most_common_items_of_user = {k: v for k, v in Counter(positive_items_per_user[user]).most_common()}
        user_output = list(most_common_items_of_user.keys())[:12]
        user_output = user_output + list(pop_items[:12 - len(user_output)])
        outputs.append(user_output)
    else:
        cnt += 1
        user_output = list(pop_items[:12])
        outputs.append(user_output)
print(cnt)
print("mAP Score on Validation set:", mapk(val_items, outputs))


# In[13]:


with Timer(prefix="load sub"):
    sub = pd.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')


# In[14]:


output = []
cnt = 0

for user in tqdm(sub["customer_id"]):
    if user in positive_items_per_user.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user[user]).most_common()}
        user_output = list(most_common_items_of_user.keys())[:12]
        user_output = user_output + list(pop_items[:12 - len(user_output)])
        output.append(user_output)
    else:
        cnt += 1
        user_output = list(pop_items[:12])
        output.append(user_output)

str_outputs = []
print(len(output))
print(cnt)
for out in output:
    str_outputs.append(" ".join([str(x) for x in out]))


# In[15]:


sub["prediction"] = str_outputs
sub.to_csv("submission.csv", index=False)
sub.head()


# In[ ]:





# In[ ]:




