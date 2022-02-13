#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from typing import Union, Tuple, Optional, List
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
    exp_name = "eda"
    input_dir = f"../input/h-and-m-personalized-fashion-recommendations/"
    output_dir = f"./" 
    model_dir = f"./"
    log_dir = f"./"
    seed = 42


# In[3]:


from utils.logger import Logger
from utils.reduce_mem_usage import reduce_mem_usage
from utils.timer import Timer
from utils.util import decorate
logger = Logger(path=Config.log_dir, exp_name=Config.exp_name)


# In[4]:


with Timer(logger=logger, prefix="loading articles.csv"):
    art_df = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/articles.csv")
    logger.info(decorate("art_df.shape"))
    logger.info(art_df.shape)
    
with Timer(logger=logger, prefix="loading custmer.csv"):
    customer_df = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/customers.csv")
    logger.info(decorate("customer_df.shape"))
    logger.info(customer_df.shape)
    
with Timer(logger=logger, prefix="loading train.csv"):
    train_df = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")
    logger.info(decorate("train_df.shape"))
    logger.info(train_df.shape)
    
with Timer(logger=logger, prefix="loading sample_submission.csv"):
    sub_df = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv")
    logger.info(decorate("sub_df.shape"))
    logger.info(sub_df.shape)


# In[5]:


print(art_df.info())


# In[6]:


print(customer_df.info())


# In[7]:


print(train_df.info())


# In[8]:


print(sub_df.info())


# In[9]:


with Timer(logger=logger, prefix="reduce_mem_usage"):
    art_df = reduce_mem_usage(art_df)
    print("*" * 30)
    customer_df = reduce_mem_usage(customer_df)
    print("*" * 30)
    train_df = reduce_mem_usage(train_df)


# In[10]:


with Timer(logger=logger, prefix="check isnull sum"):
    print(decorate("train_df"))
    print(train_df.isnull().sum())
    print(decorate("art_df"))
    print(art_df.isnull().sum())
    print(decorate("customer_df"))
    print(customer_df.isnull().sum())


# In[11]:


with Timer(logger=logger, prefix="check stats info"):
    print(decorate("train stats info"))
    print(train_df.describe())
    print(decorate("customer stats info"))
    print(customer_df.describe())
    
print(decorate("art stats info"))
art_df.describe()


# In[12]:


train_df.head()


# In[13]:


customer_df.head()


# In[14]:


art_df.head()


# In[15]:


with Timer(logger=logger, prefix="check unique value of art df"):
    print(decorate("prod_name"))
    print(art_df["prod_name"].nunique())
    print(decorate("product_type_name"))
    print(art_df["product_type_name"].nunique())
    print(decorate("product_group_name"))
    print(art_df["product_group_name"].nunique())
    print(decorate("graphical_appearance_name"))
    print(art_df["graphical_appearance_name"].nunique())
    print(decorate("colour_group_name"))
    print(art_df["colour_group_name"].nunique())
    print(decorate("perceived_colour_value_name"))
    print(art_df["perceived_colour_value_name"].nunique())
    print(decorate("perceived_colour_master_name"))
    print(art_df["perceived_colour_master_name"].nunique())
    print(decorate("department_nam"))
    print(art_df["department_name"].nunique())
    print(decorate("index_name"))
    print(art_df["index_name"].nunique())
    print(decorate("index_group_name"))
    print(art_df["index_group_name"].nunique())
    print(decorate("section_name"))
    print(art_df["section_name"].nunique())
    print(decorate("garment_group_name"))
    print(art_df["garment_group_name"].nunique())
    


# In[16]:


from vis.boxplot import plot_boxplot
from vis.value_count import plot_value_count


# ## train_df

# In[17]:


with Timer(logger=logger, prefix="plot train price"):
    plt.figure(figsize=(20, 7))
    sns.distplot(train_df["price"])
    plt.show()


# In[18]:


with Timer(logger=logger, prefix="plot train price boxplot"):
    plot_boxplot(df=train_df, col_name="price")


# In[19]:


with Timer(logger=logger, prefix="plot value count train_df sales_channel_id"):
    plot_value_count(df=train_df, col_name="sales_channel_id")


# ## Customer_df

# In[20]:


with Timer(logger=logger, prefix="plot customer age distplot"):
    plt.figure(figsize=(20, 7))
    sns.distplot(customer_df["age"])
    plt.show()


# In[21]:


with Timer(logger=logger, prefix="plot customer age boxplot"):
    plot_boxplot(df=customer_df, col_name="age")


# In[22]:


with Timer(logger=logger, prefix="plot value count club member status"):
    plot_value_count(df=customer_df, col_name="club_member_status")


# In[23]:


with Timer(logger=logger, prefix="plot value count fashion_news_frequency"):
    plot_value_count(df=customer_df, col_name="fashion_news_frequency")


# ## art_df

# In[24]:


with Timer(logger=logger, prefix="value count plot product_group_name"):
    display(art_df["product_group_name"].value_counts())
    plot_value_count(df=art_df, col_name="product_group_name")


# In[25]:


with Timer(logger=logger, prefix="value count plot graphical_appearance_name"):
    display(art_df["graphical_appearance_name"].value_counts())
    plot_value_count(df=art_df, col_name="graphical_appearance_name")


# In[26]:


with Timer(logger=logger, prefix="value count plot colour_group_name"):
    display(art_df["colour_group_name"].value_counts())
    plot_value_count(df=art_df, col_name="colour_group_name")


# In[27]:


with Timer(logger=logger, prefix="value count plot perceived_colour_value_name"):
    display(art_df["perceived_colour_value_name"].value_counts())
    plot_value_count(df=art_df, col_name="perceived_colour_value_name")


# In[28]:


with Timer(logger=logger, prefix="value count plot perceived_colour_master_name"):
    display(art_df["perceived_colour_master_name"].value_counts())
    plot_value_count(df=art_df, col_name="perceived_colour_master_name")


# In[29]:


with Timer(logger=logger, prefix="value count plot department_name"):
    display(art_df["department_name"].value_counts())
    plot_value_count(df=art_df, col_name="department_name")


# In[30]:


with Timer(logger=logger, prefix="value count plot index name"):
    display(art_df["index_name"].value_counts())
    plot_value_count(df=art_df, col_name="index_name")


# In[31]:


with Timer(logger=logger, prefix="value count plot index group name"):
    display(art_df["index_group_name"].value_counts())
    plot_value_count(df=art_df, col_name="index_group_name")


# In[32]:


with Timer(logger=logger, prefix="value count plot section name"):
    display(art_df["section_name"].value_counts())
    plot_value_count(df=art_df, col_name="section_name")


# In[33]:


with Timer(logger=logger, prefix="value count plot garment_group_name"):
    display(art_df["garment_group_name"].value_counts())
    plot_value_count(df=art_df, col_name="garment_group_name")


# In[34]:


with Timer(logger=logger, prefix="value count plot product_type_name"):
    display(art_df["product_type_name"].value_counts())
    plot_value_count(df=art_df, col_name="product_type_name")


# In[35]:


def preprocessing_date(input_df: pd.DataFrame) -> pd.DataFrame:
    date_df = pd.to_datetime(input_df["t_dat"])
    out_df = pd.DataFrame({
        "year":date_df.dt.year,
        "month":date_df.dt.month,
        "day":date_df.dt.day,
        "dayofweek":date_df.dt.dayofweek
    })
    return out_df

with Timer(logger=logger, prefix="preprocessing date"):
    date_df = preprocessing_date(train_df)


# ## date_df

# In[36]:


with Timer(logger=logger, prefix="value count year"):
    plot_value_count(date_df, col_name="year")


# In[37]:


with Timer(logger=logger, prefix="plot value count month"):
    plot_value_count(df=date_df, col_name="month")


# In[38]:


with Timer(logger=logger, prefix="plot value count day"):
    plot_value_count(df=date_df, col_name="day")


# In[39]:


with Timer(logger=logger, prefix="plot value count dayofweek"):
    plot_value_count(df=date_df, col_name="dayofweek")


# ## nlp_date

# In[40]:


def preprocessing_nlp(input_df: pd.DataFrame) -> pd.DataFrame:
    nlp_data = input_df[["detail_desc"]].copy()
    nlp_data["string_count"] = nlp_data["detail_desc"].apply(lambda x: len(x))
    nlp_data["word_count"] = nlp_data["detail_desc"].apply(lambda x: len(x.split()))
    return nlp_data


# In[41]:


with Timer(logger=logger, prefix="preprocess nlp"):
    nlp_data = preprocessing_nlp(art_df)


# In[42]:


nlp_data.describe()


# In[43]:


with Timer(logger=logger, prefix="string count"):
    plt.figure(figsize=(20, 7))
    sns.distplot(nlp_data["string_count"])
    plt.show()


# In[44]:


with Timer(logger=logger, prefix="word count"):
    plt.figure(figsize=(20, 7))
    sns.distplot(nlp_data["word_count"])
    plt.show()


# In[45]:


from vis.ngram_plot import create_top_ngram_word_plot
from vis.wordcloud_plot import create_wordcloud


# In[46]:


nlp_data.dropna(inplace=True)


# ## with stopwords

# In[47]:


with Timer(logger=logger, prefix="n_gram_word_plot"):
    create_top_ngram_word_plot(input_df=nlp_data, col="detail_desc")


# In[48]:


with Timer(logger=logger, prefix="create_wordcloud_plot"):
    create_wordcloud(input_df=nlp_data, col="detail_desc")


# In[49]:


import nltk
from nltk.corpus import stopwords


# ## removing stopwords

# In[50]:


with Timer(logger=logger, prefix="n_gram_word_plot"):
    create_top_ngram_word_plot(input_df=nlp_data, col="detail_desc", stop_words=stopwords.words('english'))


# In[51]:


with Timer(logger=logger, prefix="create_wordcloud_plot"):
    create_wordcloud(input_df=nlp_data, col="detail_desc", stopwords=stopwords.words("english"))


# In[52]:


print(train_df.shape)
train_df.head()


# In[53]:


print(customer_df.shape)
customer_df.head()


# In[54]:


print(art_df.shape)
art_df.head()


# In[55]:


from vis.venn_plot import plot_intersection


# In[56]:


print(train_df["customer_id"].nunique())
print(customer_df["customer_id"].nunique())
plot_intersection(train_df, customer_df, "customer_id", "customer_id")


# In[57]:


print(train_df["article_id"].nunique())
print(art_df["article_id"].nunique())
plot_intersection(train_df, art_df, "article_id", "article_id")


# ## pd.merge vs pd.join

# In[58]:


with Timer(logger=logger, prefix="pd.merge train_df, customer_df"):
    pd.merge(train_df, customer_df, on=["customer_id"], how="left")


# In[59]:


with Timer(logger=logger, prefix="pd.merge train_df art_df"):
    pd.merge(train_df, art_df, on=["article_id"], how="left")


# In[60]:


with Timer(logger=logger, prefix="pd.join train_df customer_df"):
    train_df.join(customer_df.set_index("customer_id"), how="left", on="customer_id")


# In[61]:


with Timer(logger=logger, prefix="pd.join train_df art_df"):
    train_df.join(art_df.set_index("article_id"), how="left", on="article_id")


# In[62]:


with Timer(logger=logger, prefix="pd.join train_df art_df customer_df"):
    train_df = train_df.join(customer_df.set_index("customer_id"), how="left", on="customer_id")
    train_df = train_df.join(art_df.set_index("article_id"), how="left", on="article_id")


# In[63]:


transaction = pd.read_csv("../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv")
transaction = reduce_mem_usage(transaction)


# ## Store df

# In[64]:


from utils.util import Util


# In[65]:


with Timer(logger, prefix="store pickle"):
    Util.dump_df(train_df, "train.pkl", is_pickle=True)
    Util.dump_df(art_df, "art.pkl", is_pickle=True)
    Util.dump_df(customer_df, "customer.pkl", is_pickle=True)
    Util.dump_df(transaction, "transaction.pkl", is_pickle=True)


# In[66]:


import gc
del customer_df, transaction, art_df
gc.collect()

