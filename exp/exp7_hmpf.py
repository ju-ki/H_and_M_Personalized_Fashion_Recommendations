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
    exp_name = "exp6"
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


from tqdm.notebook import tqdm


# In[8]:


def training(start_datetime, end_datetime):
    train = transaction_df.loc[(transaction_df["t_dat"] >= start_datetime) & (transaction_df["t_dat"] < end_datetime), :]
    valid = transaction_df.loc[transaction_df["t_dat"] > datetime(2020, 9, 15), :]
    positive_items_per_user = train.groupby(["customer_id"])["article_id"].apply(list)
    positive_items_val = valid.groupby(["customer_id"])["article_id"].apply(list)
    val_users = positive_items_val.keys()
    val_items = []

    for i, user in tqdm(enumerate(val_users)):
        val_items.append(positive_items_val[user])
    print("Total users in validation", len(val_users))
    train["pop_factor"] = train["t_dat"].apply(lambda x: 1 / (datetime(2020, 9 ,16) - x).days)
    pop_items_group = train.groupby(["article_id"])["pop_factor"].sum()
    _, pop_items = zip(*sorted(zip(pop_items_group, pop_items_group.keys()))[::-1])

    outputs = []
    cnt = 0

    #9月 + 8月限定で見たときのレコメンド
    for i, user in tqdm(enumerate(val_users)):
        #8月9月に含まれていないやつに関しては9月の期間のtop12
        if user not in positive_items_per_user.keys():
            cnt += 1
            outputs.append(pop_items[:12])
            continue
        if user in positive_items_per_user.keys():
            most_common_items_of_user = {k: v for k, v in Counter(positive_items_per_user[user]).most_common()}
            user_output = list(most_common_items_of_user.keys())[:12]
        # 8月の期間
#             else:
#                 most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user_aug[user]).most_common()}
#                 user_output = list(most_common_items_of_user.keys())[:4]
        user_output = user_output + list(pop_items[:12 - len(user_output)])
        outputs.append(user_output)

    print(cnt)
    print("mAP Score on Validation set:", mapk(val_items, outputs))


# In[9]:


train1 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 9, 8)) & (transaction_df["t_dat"] < datetime(2020, 9, 16)), :]
train2 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 9, 1)) & (transaction_df["t_dat"] < datetime(2020, 9, 8)), :]
train3 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 23)) & (transaction_df["t_dat"] < datetime(2020, 9, 1)), :]
train4 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 15)) & (transaction_df["t_dat"] < datetime(2020, 8, 23)), :]
# train5 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 8)) & (transaction_df["t_dat"] < datetime(2020, 8, 15)), :]


positive_items_per_user1 = train1.groupby(["customer_id"])["article_id"].apply(list)
positive_items_per_user2 = train2.groupby(["customer_id"])["article_id"].apply(list)
positive_items_per_user3 = train3.groupby(["customer_id"])["article_id"].apply(list)
positive_items_per_user4 = train4.groupby(["customer_id"])["article_id"].apply(list)
# positive_items_per_user5 = train5.groupby(["customer_id"])["article_id"].apply(list)


valid = transaction_df.loc[(transaction_df["t_dat"] > datetime(2020, 9, 15)), :]
positive_items_val = valid.groupby(["customer_id"])["article_id"].apply(list)


# In[10]:


from tqdm.notebook import tqdm
val_users = positive_items_val.keys()
val_items = []

for i, user in tqdm(enumerate(val_users)):
    val_items.append(positive_items_val[user])
print("Total users in validation", len(val_users))


# In[11]:


train = pd.concat([train1, train2], axis=0)

train["pop_factor"] = train["t_dat"].apply(lambda x: 1 / (datetime(2020, 9 ,16) - x).days)
pop_items_group = train.groupby(["article_id"])["pop_factor"].sum()
_, pop_items = zip(*sorted(zip(pop_items_group, pop_items_group.keys()))[::-1])


# In[12]:


outputs = []
cnt = 0

for user in tqdm(val_users):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
    if user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
    if user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
    if user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
#     if user in positive_items_per_user5.keys():
#         most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user5[user]).most_common()}
#         user_output += list(most_common_items_of_user.keys())[:12]
    user_output += list(pop_items[:12 - len(user_output)])
    outputs.append(user_output)
    
print("mAP Score on Validation set:", mapk(val_items, outputs))


# In[13]:


submission = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")
submission.head()


# In[14]:


train1 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,16)) & (transaction_df['t_dat'] < datetime(2020,9,23))]
train2 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,8)) & (transaction_df['t_dat'] < datetime(2020,9,16))]
train3 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,1)) & (transaction_df['t_dat'] < datetime(2020,9,8))]
train4 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,8,23)) & (transaction_df['t_dat'] < datetime(2020,9,1))]

positive_items_per_user1 = train1.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user2 = train2.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user3 = train3.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user4 = train4.groupby(['customer_id'])['article_id'].apply(list)


# In[15]:


train = pd.concat([train1, train2, train3], axis=0)

train["pop_factor"] = train["t_dat"].apply(lambda x: 1 / (datetime(2020, 9 ,23) - x).days)
pop_items_group = train.groupby(["article_id"])["pop_factor"].sum()
_, pop_items = zip(*sorted(zip(pop_items_group, pop_items_group.keys()))[::-1])


# In[16]:


outputs = []
cnt = 0

for user in tqdm(submission['customer_id']):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12]
    if user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12 - len(user_output)]
    if user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12 - len(user_output)]
    if user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        user_output += list(most_common_items_of_user.keys())[:12 - len(user_output)]
    
    user_output += list(pop_items[:12 - len(user_output)])
    outputs.append(user_output)
    
str_outputs = []
for output in outputs:
    str_outputs.append(" ".join([str(x) for x in output]))


# In[17]:


submission["prediction"] = str_outputs
submission.to_csv("submission.csv", index=False)
submission.head()


# In[ ]:




