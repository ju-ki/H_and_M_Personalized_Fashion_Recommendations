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
    exp_name = "exp5"
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
    transaction_df = Util.load_df(Config.my_input_dir + "transaction.pkl", is_pickle=True)
    customer_df = Util.load_df(Config.my_input_dir + "customer.pkl", is_pickle=True)
    art_df = Util.load_df(Config.my_input_dir + "art.pkl", is_pickle=True)
    train_df = Util.load_df(Config.my_input_dir + "train.pkl", is_pickle=True)
transaction_df["t_dat"] = pd.to_datetime(transaction_df["t_dat"].astype(object))


# In[6]:


df_3_week = transaction_df[transaction_df["t_dat"] >= pd.to_datetime("2020-08-31")].copy()
df_2_week = transaction_df[transaction_df["t_dat"] >= pd.to_datetime("2020-09-07")].copy()
df_1_week = transaction_df[transaction_df["t_dat"] >= pd.to_datetime("2020-09-15")].copy()


# In[7]:


from tqdm.notebook import tqdm


# In[8]:


purchase_dict3_week = {}

for i, x in tqdm(enumerate(zip(df_3_week["customer_id"], df_3_week["article_id"]))):
    cust_id, art_id = x
    if cust_id not in purchase_dict3_week:
        purchase_dict3_week[cust_id] = {}
    if art_id not in purchase_dict3_week[cust_id]:
        purchase_dict3_week[cust_id][art_id] = 0
    purchase_dict3_week[cust_id][art_id] += 1
    
print(len(purchase_dict3_week))
#df3_weekで人気の商品
dummy_list3_week = list((df_3_week["article_id"].value_counts()).index)[:12]


# In[9]:


dummy_list3_week


# In[10]:


purchase_dict2_week = {}

for i, x in tqdm(enumerate(zip(df_2_week["customer_id"], df_2_week["article_id"]))):
    cust_id, art_id = x
    if cust_id not in purchase_dict2_week:
        purchase_dict2_week[cust_id] = {}
    if art_id not in purchase_dict2_week[cust_id]:
        purchase_dict2_week[cust_id][art_id] = 0
    purchase_dict2_week[cust_id][art_id] += 1
    
print(len(purchase_dict2_week))
#df2_weekで人気の商品
dummy_list2_week = list((df_2_week["article_id"].value_counts()).index)[:12]


# In[11]:


dummy_list2_week


# In[12]:


purchase_dict1_week = {}

for i, x in tqdm(enumerate(zip(df_1_week["customer_id"], df_1_week["article_id"]))):
    cust_id, art_id = x
    if cust_id not in purchase_dict1_week:
        purchase_dict1_week[cust_id] = {}
    if art_id not in purchase_dict1_week[cust_id]:
        purchase_dict1_week[cust_id][art_id] = 0
    purchase_dict1_week[cust_id][art_id] += 1
    
print(len(purchase_dict1_week))
#df1_weekで人気の商品
dummy_list1_week = list((df_1_week["article_id"].value_counts()).index)[:12]


# In[13]:


sub_df = pd.read_csv(Config.input_dir + "sample_submission.csv")
print(sub_df.head())


# In[14]:


need_improvement_model = sub_df[["customer_id"]]
prediction_list = []

dummy_pred = " ".join(dummy_list2_week)

for i, cust_id in tqdm(enumerate(sub_df["customer_id"].values.reshape((-1,)))):
#     if i > 100:
#         break
    if cust_id in purchase_dict1_week:
        # 特定の期間に買った商品を取ってきて12個に満たないなら特定の期間の人気商品を取ってくるみたい
        l = sorted((purchase_dict1_week[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l) > 12:
            s = " ".join(l[:12])
        else:
            s = " ".join(l + dummy_list1_week[:(12-len(l))])
    elif cust_id in purchase_dict2_week:
        # 特定の期間に買った商品を取ってきて12個に満たないなら特定の期間の人気商品を取ってくるみたい
        l = sorted((purchase_dict2_week[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l) > 12:
            s = " ".join(l[:12])
        else:
            s = " ".join(l + dummy_list2_week[:(12-len(l))])
    elif cust_id in purchase_dict3_week:
        # 特定の期間に買った商品を取ってきて12個に満たないなら特定の期間の人気商品を取ってくるみたい
        l = sorted((purchase_dict3_week[cust_id]).items(), key=lambda x: x[1], reverse=True)
        l = [y[0] for y in l]
        if len(l) > 12:
            s = " ".join(l[:12])
        else:
            s = " ".join(l + dummy_list3_week[:(12-len(l))])
    prediction_list.append(s)


# In[15]:


need_improvement_model["prediction"] = prediction_list
need_improvement_model.head()


# In[16]:


need_improvement_model.to_csv("submission.csv", index=False)


# In[ ]:




