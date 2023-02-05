#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


import os


# In[9]:


os.getcwd()


# In[10]:


os.chdir('C:\\Users\\Gupta Ji\\Downloads')


# In[14]:


df =pd.read_csv("HousePrices.csv")


# In[15]:


df.shape


# In[16]:


df.head()


# In[18]:


# getting the null values
df.isnull().sum().sort_values(ascending = False)


# In[19]:


#checking how much percent of that column is missing
df.isnull().mean().sort_values(ascending =False)


# In[20]:


#cheking correlation of SalePrice with other columns
corr_ser = df.corr().iloc[-1,:].sort_values(ascending = False)
corr_ser


# In[22]:


#selecting top 10 predictors
columns = corr_ser.index[:10]
columns


# In[23]:


df2 = df.loc[:,columns]


# In[24]:


df2.head()


# In[25]:


df2.isna().sum()


# In[26]:


#seperating X and y
X = df2.iloc[:,1:].values
y = df2.iloc[:,0].values


# In[27]:


#splitting train and test values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state =0)


# In[29]:


from sklearn.linear_model import Lasso
model = Lasso(alpha = 0.1,normalize = True)


# In[30]:


model.fit(X_train,y_train)


# In[31]:


#prediction
y_pred = model.predict(X_test)


# In[32]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:




