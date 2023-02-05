#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir('C:\\Users\\Gupta Ji\\Downloads')


# In[5]:


df = pd.read_csv('HousePrices.csv')


# In[6]:


df.shape


# In[7]:


df.head()


# In[10]:


#getting the null values
df.isnull().sum().sort_values(ascending = False)


# In[12]:


#checking how much percent of that column is missing
df.isnull().mean().sort_values(ascending = False)


# In[13]:


#checking correlation of SalePrice with other columns
corr_ser = df.corr().iloc[-1,:].sort_values(ascending = False)
corr_ser


# In[14]:


#selecting top 10 predictors
columns = corr_ser.index[:10]
columns


# In[15]:


df2 =df.loc[:,columns]


# In[16]:


df2.head()


# In[17]:


df2.isna().sum()


# In[18]:


#sepretting X and y
X = df2.iloc[:,1:].values
y =df2.iloc[:,0].values


# In[19]:


#splitting train and test values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state =0)


# In[20]:


from sklearn.linear_model import Ridge


# In[21]:


model = Ridge(normalize = True)
model.fit(X_train,y_train)


# In[22]:


#prediction
y_pred =model.predict(X_test)


# In[23]:


from sklearn.metrics import r2_score


# In[24]:


r2_score(y_test,y_pred)


# In[ ]:




