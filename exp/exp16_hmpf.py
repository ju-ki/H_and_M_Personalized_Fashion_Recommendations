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
import datetime
from tqdm.auto import tqdm
from collections import Counter, defaultdict
from typing import Union, Tuple, Optional, List

import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
plt.style.use("ggplot")
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


# In[2]:


def iter_to_str(iterable):
    return " ".join(map(lambda x: str(0) + str(x), iterable))

def blend(dt, w=[], k=12):
    if len(w) == 0:
        w = [1] * (len(dt))
    preds = []
    for i in range(len(w)):
        preds.append(dt[i].split())
    res = {}
    for i in range(len(preds)):
        if w[i] < 0:
            continue
        for n, v in enumerate(preds[i]):
            if v in res:
                res[v] += (w[i] / (n + 1))
            else:
                res[v] = (w[i] / (n + 1))    
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return ' '.join(res[:k])

def prune(pred, ok_set, k=12):
    pred = pred.split()
    post = []
    for item in pred:
        if int(item) in ok_set and not item in post:
            post.append(item)
    return " ".join(post[:k])

def apk(actual, predicted, k=12):
    if len(predicted) > k:
        predicted = predicted[:k]
    score, nhits = 0.0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            nhits += 1.0
            score += nhits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=12, return_apks=False):
    assert len(actual) == len(predicted)
    apks = [apk(ac, pr, k) for ac, pr in zip(actual, predicted) if 0 < len(ac)]
    if return_apks:
        return apks
    return np.mean(apks)

def validation(actual, predicted, grouping, score=0, index=-1, ignore=False, figsize=(12, 6)):
    # actual, predicted : list of lists
    # group : pandas Series
    # score : pandas DataFrame
    if ignore: return
    ap12 = mapk(actual, predicted, return_apks=True)
    map12 = round(np.mean(ap12), 6)
    if isinstance(score, int): score = pd.DataFrame({g:[] for g in sorted(grouping.unique().tolist())})
    if index == -1 : index = score.shape[0]
    score.loc[index, "All"] = map12
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1); sns.histplot(data=ap12, log_scale=(0, 10), bins=20); plt.title(f"MAP@12 : {map12}")
    for g in grouping.unique():
        map12 = round(mapk(actual[grouping == g], predicted[grouping == g]), 6)
        score.loc[index, g] = map12
    plt.subplot(1, 2, 2); score[[g for g in grouping.unique()[::-1]] + ['All']].loc[index].plot.barh(); plt.title(f"MAP@12 of Groups")
    vc = pd.Series(predicted).apply(len).value_counts()
    score.loc[index, "Fill"] = round(1 - sum(vc[k] * (12 - k) / 12 for k in (set(range(12)) & set(vc.index))) / len(actual), 3) * 100
    display(score)
    return score


# In[3]:


transaction_df = pd.read_parquet("../input/hm-parquets-of-datasets/transactions_train.parquet")
sub_df = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")
customer_id = pd.DataFrame(sub_df.customer_id.apply(lambda s: int(s[-16:], 16)))
customer_df = pd.read_parquet("../input/hm-parquets-of-datasets/customers.parquet")
article_df = pd.read_parquet("../input/hm-parquets-of-datasets/articles.parquet")
print(transaction_df.shape, sub_df.shape, customer_df.shape, article_df.shape)


# In[4]:


group = transaction_df.groupby("customer_id").sales_channel_id.mean().round().reset_index()              .merge(customer_id, on="customer_id", how="right").rename(columns={"sales_channel_id": "group"})
grouping = group.group.fillna(1.0)


# In[5]:


test_week = 104
# id of week to be used in a validation; set 105 if you would like to create a submission
test = transaction_df.loc[transaction_df.week == test_week].groupby('customer_id').article_id.apply(iter_to_str).reset_index()    .merge(customer_id, on='customer_id', how='right')
test_actual = test.article_id.apply(lambda s: [] if pd.isna(s) else s.split())
last_date = transaction_df.loc[transaction_df.week < test_week].t_dat.max()


# In[6]:


init_date = last_date - datetime.timedelta(days=9999)
train_df = transaction_df.loc[transaction_df.t_dat <= last_date].copy()
#最後にいつ買ったか(l_dat)
train_df = train_df.merge(train_df.groupby('customer_id').t_dat.max().reset_index().rename(columns={'t_dat':'l_dat'}),
                   on = 'customer_id', how='left')
train_df["d_dat"] = (train_df.l_dat - train_df.t_dat).dt.days
train_df = train_df.loc[train_df.d_dat < 14].sort_values(["t_dat"], ascending=False).drop_duplicates(["customer_id", "article_id"])
#最後に買ったアイテムからsubmissionと紐づいているcustomer_idのarticle_idを取得

sub_df["last_purchase"] = train_df.groupby("customer_id").article_id.progress_apply(iter_to_str).reset_index().merge(customer_id, on="customer_id", how="right").article_id.fillna("")
predicted = sub_df['last_purchase'].progress_apply(lambda s: [] if pd.isna(s) else s.split())
score = validation(test_actual, predicted, grouping, index='Last Purchase', ignore=(test_week == 105))


# In[7]:


init_date = last_date - datetime.timedelta(days=6)
train_df = transaction_df[(transaction_df.t_dat >= init_date) & (transaction_df.t_dat <= last_date)].copy()          .groupby(["article_id"]).t_dat.count().reset_index()
article_df = article_df.merge(train_df, on='article_id', how='left').rename(columns={'t_dat':'ct'})    .sort_values('ct', ascending=False).query('ct > 0')

# product codeはarticle_id上6桁 -> productは同じだが他が違う(今回は色)
map_to_col = defaultdict(list)
for aid in tqdm(article_df.article_id.tolist()):
    map_to_col[aid] = list(filter(lambda x: x!= aid, article_df[article_df.product_code == aid // 1000].article_id.tolist()))[:1]
    
def map_to_variation(s):
    f = lambda item: iter_to_str(map_to_col[int(item)])
    return " ".join(map(f, s.split()))

sub_df["other_colors"] = sub_df["last_purchase"].fillna("").progress_apply(map_to_variation)
predicted = sub_df["other_colors"].progress_apply(lambda x: [] if pd.isna(x) else x.split())
score = validation(test_actual, predicted, grouping, score, index="Other Colors", ignore=(test_week == 105))


# In[8]:


# init_date = last_date - datetime.timedelta(days=5-1)
# group_df = pd.concat([customer_id, group.group.fillna(1)], axis=1)
# group_df.columns = ["customer_id", "group"]
# train_df = transaction_df.loc[(transaction_df.t_dat >= init_date) & (transaction_df.t_dat <= last_date)].copy()\
#         .merge(group_df, on="customer_id", how="left")\
#         .groupby(["group", "article_id"]).t_dat.count().reset_index()

# items = defaultdict(str)
# for g in train_df.group.unique():
#     items[g] = iter_to_str(train_df.loc[train_df.group == g].sort_values('t_dat', ascending=False).article_id.tolist()[:12])
# sub_df["popular_items"] = group_df.group.map(items)
# predicted = sub_df["popular_items"].progress_apply(lambda x : [] if pd.isna(x) else x.split())
# score = validation(test_actual, predicted, grouping, score, index="Popular Items", ignore=(test_week == 105))


# In[9]:


init_date = last_date - datetime.timedelta(days=5-1)
age_bins = [-1, 19, 29, 39, 49, 59, 69, 119]
customer_df["age_bin"] = pd.cut(customer_df["age"], age_bins, labels=[0, 1, 2, 3, 4, 5, 6])
age_df = pd.concat([customer_id, customer_df["age_bin"].fillna(0)], axis=1)
train_df = transaction_df.loc[(transaction_df.t_dat >= init_date) & (transaction_df.t_dat <= last_date)].copy()        .merge(age_df, on="customer_id", how="left").groupby(["age_bin", "article_id"]).t_dat.count().reset_index()
age_items = defaultdict(str)
for g in train_df.age_bin.unique():
    age_items[g] = iter_to_str(train_df.loc[train_df.age_bin == g].sort_values('t_dat', ascending=False).article_id.tolist()[:12])
sub_df["age_items"] = age_df.age_bin.map(age_items).astype(object)
predicted = sub_df["age_items"].progress_apply(lambda x : x.split())
score = validation(test_actual, predicted, grouping, score, index="Age Items", ignore=(test_week == 105))


# In[10]:


init_date = last_date - datetime.timedelta(days=11)
sold_set = set(transaction_df[(transaction_df.t_dat >= init_date) & (transaction_df.t_dat <= last_date)].article_id.tolist())
sub_df["prediction"] = sub_df[["last_purchase", "other_colors", "age_items"]]                 .progress_apply(blend, w=[100, 10, 1], axis=1, k=32).apply(prune, ok_set=sold_set)

predicted = sub_df.prediction.progress_apply(lambda x: [] if pd.isna(x) else x.split())
score = validation(test_actual, predicted, grouping, score, index="Prediction", ignore=test_week == 105)


# In[11]:


sub_df[["customer_id", "prediction"]].to_csv("submission.csv", index=False)
display(sub_df.head())

