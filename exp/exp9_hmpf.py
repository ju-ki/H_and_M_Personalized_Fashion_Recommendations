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
    exp_name = "exp9"
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
    articles_df = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
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


def get_alternate_most_popular(df_data, factor, return_orig=False):
    
    next_best_match = []
    
    df = df_data.copy()
    df['article_count'] = df.groupby('article_id')['customer_id'].transform('count')
    df['article_min_price'] = df.groupby('article_id')['price'].transform('min')
    count_df = df[['article_id', 'article_count', 'article_min_price']].drop_duplicates().reset_index(drop=True)
    
    del df
    
    for article in tqdm(count_df.article_id.tolist()):
        prodname = articles_df[articles_df.article_id==int(article)]['prod_name'].iloc[0]
        other_article_list = articles_df[articles_df.prod_name==prodname]['article_id'].tolist()
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
    next_best_price = []
    for article in tqdm(right_df['article_id']):
        if len(count_df[count_df.article_id==article]['article_count'])>0:
            next_best_count.append(count_df[count_df.article_id==article]['article_count'].iloc[0])
            next_best_price.append(count_df[count_df.article_id==article]['article_min_price'].iloc[0])
        else:
            next_best_count.append(0)
            next_best_price.append(0)

    count_df['count_next_best'] = next_best_count
    count_df['next_best_min_price'] = next_best_price
        
    more_popular_alternatives = count_df[(count_df.article_min_price >= count_df.next_best_min_price) & 
                                         (count_df.count_next_best > factor *count_df.article_count)].copy().reset_index(drop=True)
    more_popular_alt_list = more_popular_alternatives.article_id.unique().tolist()
    
    if return_orig:
        return more_popular_alt_list, more_popular_alternatives, count_df
    else:
        return more_popular_alt_list, more_popular_alternatives


# In[9]:


train1 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 9, 8)) & (transaction_df["t_dat"] < datetime(2020, 9, 16)), :]
train2 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 9, 1)) & (transaction_df["t_dat"] < datetime(2020, 9, 8)), :]
train3 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 23)) & (transaction_df["t_dat"] < datetime(2020, 9, 1)), :]
train4 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 15)) & (transaction_df["t_dat"] < datetime(2020, 8, 23)), :]
train5 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020, 8, 8)) & (transaction_df["t_dat"] < datetime(2020, 8, 15)), :]


positive_items_per_user1 = train1.groupby(["customer_id"])["article_id"].apply(list)
positive_items_per_user2 = train2.groupby(["customer_id"])["article_id"].apply(list)
positive_items_per_user3 = train3.groupby(["customer_id"])["article_id"].apply(list)
positive_items_per_user4 = train4.groupby(["customer_id"])["article_id"].apply(list)
# positive_items_per_user5 = train5.groupby(["customer_id"])["article_id"].apply(list)


valid = transaction_df.loc[(transaction_df["t_dat"] > datetime(2020, 9, 15)), :]
positive_items_val = valid.groupby(["customer_id"])["article_id"].apply(list)


# In[10]:


alt_list_1v, alt_df_1v = get_alternate_most_popular(train2, 2, return_orig=False)
alt_list_2v, alt_df_2v = get_alternate_most_popular(train3, 2, return_orig=False)
alt_list_3v, alt_df_3v = get_alternate_most_popular(train4, 2, return_orig=False)
alt_list_4v, alt_df_4v = get_alternate_most_popular(train5, 2, return_orig=False)


# In[11]:


from tqdm.notebook import tqdm
val_users = positive_items_val.keys()
val_items = []

for i, user in tqdm(enumerate(val_users)):
    val_items.append(positive_items_val[user])
print("Total users in validation", len(val_users))


# In[12]:


train = pd.concat([train1, train2], axis=0)

train["pop_factor"] = train["t_dat"].apply(lambda x: 1 / (datetime(2020, 9 ,16) - x).days)
pop_items_group = train.groupby(["article_id"])["pop_factor"].sum()
_, pop_items = zip(*sorted(zip(pop_items_group, pop_items_group.keys()))[::-1])


# In[13]:


outputs = []
cnt = 0

for i, user in tqdm(enumerate(val_users)):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_1v:
                al.append(alt_df_1v[alt_df_1v.article_id==l[j]]["next_best_article"].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_2v:
                al.append(alt_df_2v[alt_df_2v.article_id==l[j]]["next_best_article"].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_3v:
                al.append(alt_df_3v[alt_df_3v.article_id==l[j]]["next_best_article"].iloc[0])
        l = l + al
        user_output += l[:12]
    if user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_4v:
                al.append(alt_df_4v[alt_df_4v.article_id==l[j]]["next_best_article"].iloc[0])
        l = l + al
        user_output += l[:12]

    user_output += list(pop_items[:12 - len(user_output)])
    outputs.append(user_output)
    
print("mAP Score on Validation set:", mapk(val_items, outputs))


# In[14]:


submission = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")
submission.head()


# In[15]:


train1 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,16)) & (transaction_df['t_dat'] < datetime(2020,9,23))]
train2 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,8)) & (transaction_df['t_dat'] < datetime(2020,9,16))]
train3 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,9,1)) & (transaction_df['t_dat'] < datetime(2020,9,8))]
train4 = transaction_df.loc[(transaction_df["t_dat"] >= datetime(2020,8,23)) & (transaction_df['t_dat'] < datetime(2020,9,1))]

positive_items_per_user1 = train1.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user2 = train2.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user3 = train3.groupby(['customer_id'])['article_id'].apply(list)
positive_items_per_user4 = train4.groupby(['customer_id'])['article_id'].apply(list)


# In[16]:


train = pd.concat([train1, train2, train3], axis=0)

train["pop_factor"] = train["t_dat"].apply(lambda x: 1 / (datetime(2020, 9 ,23) - x).days)
pop_items_group = train.groupby(["article_id"])["pop_factor"].sum()
_, pop_items = zip(*sorted(zip(pop_items_group, pop_items_group.keys()))[::-1])


# In[17]:


alt_list_1, alt_df_1 = get_alternate_most_popular(train2, 2, return_orig=False)
alt_list_2, alt_df_2 = alt_list_1v, alt_df_1v
alt_list_3, alt_df_3 = alt_list_2v, alt_df_2v
alt_list_4, alt_df_4 = alt_list_3v, alt_df_3v


# In[18]:


outputs = []
cnt = 0

for user in tqdm(submission['customer_id']):
    user_output = []
    if user in positive_items_per_user1.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user1[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_1:
                al.append(alt_df_1[alt_df_1.article_id==l[j]]["next_best_article"].iloc[0])
        l = l + al
        user_output += l[:12]
    if user in positive_items_per_user2.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user2[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_2:
                al.append(alt_df_2[alt_df_2.article_id==l[j]]["next_best_article"].iloc[0])
        l = l + al
        user_output += l[:12]
    if user in positive_items_per_user3.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user3[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_3:
                al.append(alt_df_3[alt_df_3.article_id==l[j]]["next_best_article"].iloc[0])
        l = l + al
        user_output += l[:12]
        
    if user in positive_items_per_user4.keys():
        most_common_items_of_user = {k:v for k, v in Counter(positive_items_per_user4[user]).most_common()}
        l = list(most_common_items_of_user.keys())
        al = []
        for j in range(0, len(l)):
            if l[j] in alt_list_4:
                al.append(alt_df_4[alt_df_4.article_id==l[j]]["next_best_article"].iloc[0])
        l = l + al
        user_output += l[:12]
    
    user_output += list(pop_items[:12 - len(user_output)])
    outputs.append(user_output)
    
str_outputs = []
for output in outputs:
    str_outputs.append(" ".join([str(x) for x in output]))


# In[19]:


submission["prediction"] = str_outputs
submission.to_csv("submission.csv", index=False)
submission.head()


# In[ ]:




