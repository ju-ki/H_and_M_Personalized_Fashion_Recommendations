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


from reco.recommender import FunkSVD
from reco.metrics import rmse


# In[4]:


class Config:
    competition_name = "h-and-m-personalized-fashion-recommendations"
    exp_name = "exp13"
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


def get_alternate_most_popular(df_data, factor, return_orig=False):
    
    next_best_match = []
    
    df = df_data.copy()
    df['article_count'] = df.groupby('article_id')['customer_id'].transform('count')
    count_df = df[['article_id', 'article_count']].drop_duplicates().reset_index(drop=True)
    
    del df
    
    for article in tqdm(count_df.article_id.tolist()):
        prod_name = articles_df[articles_df.article_id==int(article)]['prod_name'].iloc[0]
        other_article_list = articles_df[articles_df.prod_name==prod_name]['article_id'].tolist()
        other_article_list.remove(int(article))
        k = len(other_article_list)
        if k==1:
            next_best_match.append(other_article_list[0])
        if k>1:
            if len(count_df[np.in1d(count_df['article_id'], other_article_list)])!=0:
                next_best_match.append(count_df[np.in1d(count_df['article_id'], other_article_list)].sort_values('article_count', ascending=False)['article_id'].iloc[0])
            else:
                next_best_match.append(np.nan)
        if k==0:
            next_best_match.append(np.nan)

    count_df['next_best_article'] = next_best_match
    count_df['next_best_article'] = count_df['next_best_article'].fillna(0).astype(int)
    count_df['next_best_article'] = np.where(count_df['next_best_article']==0, count_df['article_id'], str(0)+count_df['next_best_article'].astype(str))

    right_df = count_df[['next_best_article']].copy().rename(columns={'next_best_article':'article_id'})

    next_best_count = []
#     next_best_price = []
    for article in tqdm(right_df['article_id']):
        if len(count_df[count_df.article_id==article]['article_count'])>0:
            next_best_count.append(count_df[count_df.article_id==article]['article_count'].iloc[0])
#             next_best_price.append(count_df[count_df.article_id==article]['article_min_price'].iloc[0])
        else:
            next_best_count.append(0)
#             next_best_price.append(0)

    count_df['count_next_best'] = next_best_count
#     count_df['next_best_min_price'] = next_best_price
        
    more_popular_alternatives = count_df[count_df.count_next_best > factor *count_df.article_count].copy().reset_index(drop=True)
    more_popular_alt_list = more_popular_alternatives.article_id.unique().tolist()
    
    if return_orig:
        return more_popular_alt_list, more_popular_alternatives, count_df
    else:
        return more_popular_alt_list, more_popular_alternatives


# In[8]:


train1 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 9, 8)) & (transaction_df["t_dat"] < datetime(2020, 9, 16)), :]
train2 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 9, 1)) & (transaction_df["t_dat"] < datetime(2020, 9, 8)), :]
train3 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 23)) & (transaction_df["t_dat"] < datetime(2020, 9, 1)), :]
# train4 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 15)) & (transaction_df["t_dat"] < datetime(2020, 8, 23)), :]

train = pd.concat([train1, train2])

positive_items_train = train.groupby(["customer_id"])["article_id"].progress_apply(list)

valid = transaction_df.loc[(transaction_df["t_dat"] > datetime(2020, 9, 15)), :]
positive_items_val = valid.groupby(["customer_id"])["article_id"].progress_apply(list)


# In[9]:


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


# In[10]:


_, pop_items = zip(*sorted(zip(pop_items_group, pop_items_group.keys()))[::-1])
train_pop["pop_factor"].describe()


# In[11]:


svd = FunkSVD(k=64, learning_rate=0.008, regularizer=0.2, iterations=1000, method="stochastic", bias=True)
svd.fit(X=train, formatizer={"user":"customer_id", "item":"article_id", "value":"feedback"}, verbose=False)


# In[12]:


val_users = positive_items_val.keys()
val_items = []

for i, user in tqdm(enumerate(val_users)):
    val_items.append(positive_items_val[user])
print("Total users in validation", len(val_users))


# In[13]:


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


# In[14]:


# 0.02059815743936571 simple
# 0.02139277299171194 train2抜き
# 0.02157705290360704 (reg=0.2, k=64, iteration=1000)


# In[15]:


with Timer(logger, prefix="loading sub_df"):
    submission = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")


# In[16]:


train1 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,16)) & (transaction_df['t_dat'] < datetime(2020,9,23))]
train2 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,8)) & (transaction_df['t_dat'] < datetime(2020,9,16))]
train3 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,1)) & (transaction_df['t_dat'] < datetime(2020,9,8))]

train = pd.concat([train1, train2])

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


# In[17]:


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


# In[18]:


submission["prediction"] = str_outputs
submission.to_csv("submission.csv", index=False)
submission.head()


# In[ ]:




