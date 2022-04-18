#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
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
from tqdm.auto import tqdm
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
tqdm.pandas()

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

from utils.logger import Logger
from utils.reduce_mem_usage import reduce_mem_usage
from utils.timer import Timer
from utils.util import decorate, Util


# In[2]:


get_ipython().system('pip install -q git+https://github.com/mayukh18/reco.git')


# In[3]:


from reco.recommender import FunkSVD, FM
from reco.metrics import rmse


# In[4]:


class Config:
    competition_name = "h-and-m-personalized-fashion-recommendations"
    exp_name = "exp11"
    input_dir = f"../input/h-and-m-personalized-fashion-recommendations/"
    my_input_dir = f"../input/hm-create-dataset-samples/"
    output_dir = f"./" 
    model_dir = f"./"
    log_dir = f"./"
    seed = 42
    DEBUG = False
    
    
def seed_everything(SEED):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
seed_everything(Config.seed)

logger = Logger(path=Config.log_dir, exp_name=Config.exp_name)


# In[5]:


with Timer(logger, prefix="load csv file"):
    if Config.DEBUG:
        transaction_df = pd.read_csv(Config.my_input_dir + "transactions_train_sample5.csv.gz", dtype={"article_id":str}, parse_dates=["t_dat"])
    else:
        transaction_df = pd.read_csv(Config.input_dir + "transactions_train.csv", dtype={"article_id": str}, parse_dates=["t_dat"])
    customer_df = pd.read_csv(Config.input_dir + "customers.csv")
    articles_df = pd.read_csv(Config.input_dir + "articles.csv")
print(transaction_df.shape, customer_df.shape, articles_df.shape)


# In[6]:


def apk(actual, predicted, k=12):
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


def mapk(actual, predicted, k=12):
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


train1 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 9, 8)) & (transaction_df["t_dat"] < datetime(2020, 9, 16)), :]
train2 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 9, 1)) & (transaction_df["t_dat"] < datetime(2020, 9, 8)), :]
train3 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 23)) & (transaction_df["t_dat"] < datetime(2020, 9, 1)), :]
# train4 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 15)) & (transaction_df["t_dat"] < datetime(2020, 8, 23)), :]

train = pd.concat([train1, train2])

positive_items_train = train.groupby(["customer_id"])["article_id"].progress_apply(list)

valid = transaction_df.loc[(transaction_df["t_dat"] > datetime(2020, 9, 15)), :]
positive_items_val = valid.groupby(["customer_id"])["article_id"].progress_apply(list)


# In[8]:


train["pop_factor"] = train["t_dat"].apply(lambda x: 1 / (datetime(2020, 9, 16) - x).days)
train_pop = train.copy()
pop_items_group =train.groupby(["article_id"])["pop_factor"].sum()

items_total_count = train.groupby(["article_id"])["article_id"].count()
users_total_count = train.groupby(["customer_id"])["customer_id"].count()

train["feedback"] = 1

train = train.groupby(["customer_id", "article_id"]).sum().reset_index()

train["feedback"] = train.apply(lambda row: row["feedback"] / (items_total_count[row["article_id"]] * users_total_count[row["customer_id"]]), axis=1)
train.drop(["price", "sales_channel_id"], axis=1, inplace=True)
train["feedback"].describe()


# In[9]:


_, pop_items = zip(*sorted(zip(pop_items_group, pop_items_group.keys()))[::-1])
train_pop["pop_factor"].describe()


# In[10]:


svd = FunkSVD(k=64, learning_rate=0.008, regularizer=0.01, iterations=200, method="stochastic", bias=True)
svd.fit(X=train, formatizer={"user":"customer_id", "item":"article_id", "value":"feedback"}, verbose=False)


# In[11]:


val_users = positive_items_val.keys()
val_items = []

for i, user in tqdm(enumerate(val_users)):
    val_items.append(positive_items_val[user])
print("Total users in validation", len(val_users))


# In[12]:


outputs = []


for user in tqdm(val_users):
    user_output = []
    if user in positive_items_train.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_train[user]).most_common()}
        items = list(most_common_items_of_user.keys())[:20]
        pred_df = pd.DataFrame({"user":[user] * len(items), "item":items})
        pred_feedback = svd.predict(pred_df, formatizer={"user":"user", "item":"item"})
        new_order = {}
        for i, item in enumerate(items):
            new_order[item] = pred_feedback[i]
        user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:12]
        
    user_output += list(pop_items[:12 - len(user_output)])
    outputs.append(user_output)
    
print("mAP Score on Validation set:", mapk(val_items, outputs))


# In[13]:


# 0.02059815743936571 simple
# 0.02139277299171194 train2抜き
# 0.02157705290360704 (reg=0.2, k=64, iteration=1000)


# In[14]:


with Timer(logger, prefix="loading sub_df"):
    submission = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")


# In[15]:


train1 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,16)) & (transaction_df['t_dat'] < datetime(2020,9,23))]
train2 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,8)) & (transaction_df['t_dat'] < datetime(2020,9,16))]
train3 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,1)) & (transaction_df['t_dat'] < datetime(2020,9,8))]

train = pd.concat([train1, train2, train3])

positive_items_train = train.groupby(["customer_id"])["article_id"].progress_apply(list)

train["pop_factor"] = train["t_dat"].apply(lambda x: 1 / (datetime(2020, 9, 23) - x).days)
train_pop = train.copy()
pop_items_group =train.groupby(["article_id"])["pop_factor"].sum()

items_total_count = train.groupby(["article_id"])["article_id"].count()
users_total_count = train.groupby(["customer_id"])["customer_id"].count()

train["feedback"] = 1

train = train.groupby(["customer_id", "article_id"]).sum().reset_index()

train["feedback"] = train.apply(lambda row: row["feedback"] / (items_total_count[row["article_id"]] * users_total_count[row["customer_id"]]), axis=1)
train.drop(["price", "sales_channel_id"], axis=1, inplace=True)


# In[16]:


outputs = []
cnt = len(submission["customer_id"])


for user in tqdm(submission['customer_id']):
    user_output = []
    if user in positive_items_train.keys():
        cnt -= 1
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_train[user]).most_common()}
        items = list(most_common_items_of_user.keys())[:20]
        pred_df = pd.DataFrame({"user":[user] * len(items), "item":items})
        pred_feedback = svd.predict(pred_df, formatizer={"user":"user", "item":"item"})
        new_order = {}
        for i, item in enumerate(items):
            new_order[item] = pred_feedback[i]
        user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:12]
        
    user_output += list(pop_items[:12 - len(user_output)])
    outputs.append(user_output)
    
str_outputs = []
for output in outputs:
    str_outputs.append(" ".join([str(x) for x in output]))
print(cnt)


# In[17]:


submission["prediction"] = str_outputs
submission.to_csv("submission.csv", index=False)
submission.head()


# In[ ]:




