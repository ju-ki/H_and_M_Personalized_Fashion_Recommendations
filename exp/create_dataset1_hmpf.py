#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import pandas as pd
sys.path.append("../input/my-pipeline/code/")


# In[2]:


class Config:
    competition_name = "h-and-m-personalized-fashion-recommendations"
    exp_name = "create_dataset1"
    input_dir = f"../input/h-and-m-personalized-fashion-recommendations/"
    output_dir = f"./" 
    model_dir = f"./"
    log_dir = f"./"
    seed = 42


# In[3]:


from utils.logger import Logger
from utils.reduce_mem_usage import reduce_mem_usage
from utils.timer import Timer
from utils.util import decorate, Util
logger = Logger(path=Config.log_dir, exp_name=Config.exp_name)


# In[4]:


with Timer(logger=logger, prefix="read all csv"):
    art_df = pd.read_csv(Config.input_dir + "articles.csv", dtype={'article_id': str})
    customer_df = pd.read_csv(Config.input_dir + "customers.csv")
    transaction_df = pd.read_csv(Config.input_dir + "transactions_train.csv", dtype={'article_id': str})
    print(art_df.shape, customer_df.shape, transaction_df.shape)


# In[5]:


with Timer(logger=logger, prefix="reduce mem usage for all csv"):
    print(decorate("art_df"))
    art_df = reduce_mem_usage(art_df)
    print(decorate("customer_df"))
    customer_df = reduce_mem_usage(customer_df)
    print(decorate("transaction_df"))
    transaction_df = reduce_mem_usage(transaction_df)


# In[6]:


with Timer(logger=logger, prefix="create all dataframe"):
    train_df = transaction_df.join(customer_df.set_index("customer_id"), how="left", on="customer_id")
    train_df = train_df.join(art_df.set_index("article_id"), how="left", on="article_id")
print(train_df.shape, transaction_df.shape, customer_df.shape, art_df.shape)


# In[7]:


with Timer(logger=logger, prefix="store all files as pkl"):
    Util.dump_df(train_df, Config.output_dir+ "train.pkl", is_pickle=True)
    Util.dump_df(transaction_df, Config.output_dir+ "transaction.pkl", is_pickle=True)
    Util.dump_df(customer_df, Config.output_dir+ "customer.pkl", is_pickle=True)
    Util.dump_df(art_df, Config.output_dir+ "art.pkl", is_pickle=True)


# In[8]:


with Timer(logger=logger, prefix="load all pickle file"):
    train_df = Util.load_df(Config.output_dir + "train.pkl", is_pickle=True)
    transaction_df = Util.load_df(Config.output_dir + "transaction.pkl", is_pickle=True)
    customer_df = Util.load_df(Config.output_dir + "customer.pkl", is_pickle=True)
    art_df = Util.load_df(Config.output_dir + "art.pkl", is_pickle=True)
print(train_df.shape, transaction_df.shape, customer_df.shape, art_df.shape)


# In[ ]:




