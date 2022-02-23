#!/usr/bin/env python
# coding: utf-8

# # Recommend Items Frequently Purchased Together
# This notebook demonstrates how recommending items that are frequently purchased together is effective. The current best scoring public notebook [here][1] recommends to customers those customers' last purchases and scores public LB 0.020. In this notebook here, we will begin with that idea and add recommending items that are frequently purchased together with a customers' previous purchaes. This notebook improves the LB and scores LB 0.021. This notebook's strategy is as follows:
# * recommend items previously purchased [idea here][1]
# * recommend items that are bought together with previous purchases [idea here][2]
# * recommend popular items [idea here][1]
# 
# [1]: https://www.kaggle.com/hengzheng/time-is-our-best-friend-v2
# [2]: https://www.kaggle.com/cdeotte/customers-who-bought-this-frequently-buy-this

# # RAPIDS cuDF
# We will use RAPIDS cuDF for fast dataframe operations

# In[1]:


import cudf
print('RAPIDS version',cudf.__version__)


# # Load Transactions, Reduce Memory
# Discussion about reducing memory is [here][1]
# 
# [1]: https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635

# In[2]:


train = cudf.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv')
train['customer_id'] = train['customer_id'].str[-16:].str.hex_to_int().astype('int64')
train['article_id'] = train.article_id.astype('int32')
train.t_dat = cudf.to_datetime(train.t_dat)
train = train[['t_dat','customer_id','article_id']]
train.to_parquet('train.pqt',index=False)
print( train.shape )
train.head()


# # Find Each Customer's Last Week of Purchases
# Our final predictions will have the row order from of our dataframe. Each row of our dataframe will be a prediction. We will create the `predictionstring` later by `train.groupby('customer_id').article_id.sum()`. Since `article_id` is a string, when we groupby sum, it will concatenate all the customer predictions into a single string. It will also create the string in the order of the dataframe. So as we proceed in this notebook, we will order the dataframe how we want our predictions ordered.

# In[3]:


tmp = train.groupby('customer_id').t_dat.max().reset_index()
tmp.columns = ['customer_id','max_dat']
train = train.merge(tmp,on=['customer_id'],how='left')
train['diff_dat'] = (train.max_dat - train.t_dat).dt.days
train = train.loc[train['diff_dat']<=6]
print('Train shape:',train.shape)
train.head()


# # (1) Recommend Most Often Previously Purchased Items
# Note that many operations in cuDF will shuffle the order of the dataframe rows. Therefore we need to sort afterward because we want the most often previously purchased items first. Because this will be the order of our predictons. Since we sort by `ct` and then `t_dat` will will recommend items that have been purchased more frequently first followed by items purchased more recently second.

# In[4]:


tmp = train.groupby(['customer_id','article_id'])['t_dat'].agg('count').reset_index()
tmp.columns = ['customer_id','article_id','ct']
train = train.merge(tmp,on=['customer_id','article_id'],how='left')
train = train.sort_values(['ct','t_dat'],ascending=False)
train = train.drop_duplicates(['customer_id','article_id'])
train = train.sort_values(['ct','t_dat'],ascending=False)
print(train.shape)
train.head()


# # (2) Recommend Items Purchased Together
# In my notebook [here][1], we compute a dictionary of items frequently purchased together. We will load and use that dictionary below. Note that we use the command `drop_duplicates` so that we don't recommend an item that the user has already bought and we have already recommended above. We will need to use Pandas for some commands because RAPIDS cuDF doesn't have two conveinent commands, (1) create new column from dictionary map of another column (2) groupby aggregate strings sum.
# 
# We concatenate these rows after the rows containing customers' previous purchases. Therefore we will recommend previous items first and then items purchased together second. Note the trick to convert a column of int32 into a prediction string (using groupby agg str sum) is from notebook [here][2]
# 
# [1]: https://www.kaggle.com/cdeotte/customers-who-bought-this-frequently-buy-this
# [2]: https://www.kaggle.com/hiroshisakiyama/recommending-items-recently-bought

# In[5]:


# USE PANDAS TO MAP COLUMN WITH DICTIONARY
import pandas as pd, numpy as np
train = train.to_pandas()
pairs = np.load('../input/hmitempairs/pairs_cudf.npy',allow_pickle=True).item()
train['article_id2'] = train.article_id.map(pairs)
train.head()


# In[6]:


# RECOMMENDATION OF PAIRED ITEMS
train2 = train[['customer_id','article_id2']].copy()
train2 = train2.loc[train2.article_id2.notnull()]
train2 = train2.drop_duplicates(['customer_id','article_id2'])
train2 = train2.rename({'article_id2':'article_id'},axis=1)
print(train2.shape)
train2.head()


# In[7]:


# CONCATENATE PAIRED ITEM RECOMMENDATION AFTER PREVIOUS PURCHASED RECOMMENDATIONS
train = train[['customer_id','article_id']]
train = pd.concat([train,train2],axis=0,ignore_index=True)
train.article_id = train.article_id.astype('int32')
train = train.drop_duplicates(['customer_id','article_id'])
print(train.shape)
train.head()


# In[8]:


# CONVERT RECOMMENDATIONS INTO SINGLE STRING
train.article_id = ' 0' + train.article_id.astype('str')
preds = cudf.DataFrame( train.groupby('customer_id').article_id.sum().reset_index() )
preds.columns = ['customer_id','prediction']
preds.head()


# # (3) Recommend Last Week's Most Popular Items
# After recommending previous purchases and items purchased together we will then recommend the 12 most popular items. Therefore if our previous recommendations did not fill up a customer's 12 recommendations, then it will be filled by popular items.

# In[9]:


train = cudf.read_parquet('train.pqt')
train.t_dat = cudf.to_datetime(train.t_dat)
train = train.loc[train.t_dat >= cudf.to_datetime('2020-09-16')]
top12 = ' 0' + ' 0'.join(train.article_id.value_counts().to_pandas().index.astype('str')[:12])
print("Last week's top 12 popular items:")
print( top12 )


# # Write Submission CSV
# We will merge our predictions onto `sample_submission.csv` and submit to Kaggle.

# In[10]:


sub = cudf.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv')
sub = sub[['customer_id']]
sub['customer_id_2'] = sub['customer_id'].str[-16:].str.hex_to_int().astype('int64')
sub = sub.merge(preds.rename({'customer_id':'customer_id_2'},axis=1),    on='customer_id_2', how='left').fillna('')
del sub['customer_id_2']
sub.prediction = sub.prediction + top12
sub.prediction = sub.prediction.str.strip()
sub.prediction = sub.prediction.str[:131]
sub.to_csv(f'submission.csv',index=False)
sub.head()


# In[ ]:




