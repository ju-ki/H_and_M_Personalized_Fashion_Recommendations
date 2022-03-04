#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install -q git+https://github.com/ju-ki/my_pipeline
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


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
from collections import Counter, defaultdict
from typing import Union, Tuple, Optional, List

import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint

# from jukijuki.utils.timer import Timer
# from jukijuki.utils.logger import Logger
# from jukijuki.utils.util import Util
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


# In[3]:


class Config:
    exp_name = "exp8"
    output_dir = "./"
    input_dir = "../input/h-and-m-personalized-fashion-recommendations/"
    data_dir = "../input/create-dataset1-hmpf/"
    seed = 42


# In[4]:


train_df = pd.read_pickle(Config.data_dir + "transaction.pkl")
train_df["t_dat"] = pd.to_datetime(train_df["t_dat"].astype(object))
train_df["article_id"] = train_df["article_id"].astype(str)
train_df["customer_id"] = train_df["customer_id"].astype(object)
print(train_df.shape)


# In[5]:


last_ts = train_df["t_dat"].max()
print(last_ts)


# In[6]:


from tqdm.auto import tqdm
tqdm.pandas()
train_df["ldbw"] = train_df["t_dat"].progress_apply(lambda d: last_ts - (last_ts - d).floor("7D"))


# In[7]:


weekly_sales = train_df.drop("customer_id", axis=1).groupby(["ldbw", "article_id"]).count()
weekly_sales = weekly_sales.rename(columns={"t_dat": "count"})


# In[8]:


weekly_sales = weekly_sales.drop(["price", "sales_channel_id"], axis=1)


# In[9]:


weekly_sales.to_csv("./weekly_sales.csv", index=False)


# In[10]:


train_df = train_df.join(weekly_sales, on=["ldbw", "article_id"])


# In[11]:


weekly_sales = weekly_sales.reset_index().set_index("article_id")
last_day = last_ts.strftime("%Y-%m-%d")


# In[12]:


train_df = train_df.join(weekly_sales.loc[weekly_sales["ldbw"] == last_day, ["count"]], on="article_id", rsuffix="_targ")


# In[13]:


del weekly_sales
gc.collect()


# In[14]:


train_df["count_targ"].fillna(0, inplace=True)
train_df["quotient"] = train_df["count_targ"] / train_df["count"]


# In[15]:


train_df.to_csv("./train2.csv", index=False)


# In[16]:


target_sales = train_df.drop("customer_id", axis=1).groupby("article_id")["quotient"].sum()
general_pred = target_sales.nlargest(12).index.tolist()
print(general_pred)


# In[17]:


pd.to_pickle(general_pred, "general_pred.pkl")


# In[18]:


purchase_dict = {}
for i, v in tqdm(enumerate(train_df.index)):
#     if i > 10:
#         break
    cust_id = train_df.at[v, "customer_id"]
    art_id = train_df.at[v, "article_id"]
    t_dat = train_df.at[v, "t_dat"]
    
    if cust_id not in purchase_dict:
        purchase_dict[cust_id] = {}
    if art_id not in purchase_dict[cust_id]:
        purchase_dict[cust_id][art_id] = 0
    x = max(1, (last_ts - t_dat).days)
    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    y = a / np.sqrt(x) + b * np.exp(-c*x) - d
    value = train_df.at[v, "quotient"] * max(0, y)
    purchase_dict[cust_id][art_id] += value


# In[19]:


sub = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")

pred_list = []
for cust_id in tqdm(sub['customer_id']):
    if cust_id in purchase_dict:
        series = pd.Series(purchase_dict[cust_id])
        series = series[series > 0]
        l = series.nlargest(12).index.tolist()
        if len(l) < 12:
            l = l + general_pred[:(12-len(l))]
    else:
        l = general_pred
    pred_list.append(' '.join(l))

sub['prediction'] = pred_list
sub.to_csv('submission.csv', index=None)
display(sub.head())


# In[ ]:




