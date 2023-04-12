#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


import os


# In[3]:


os.getcwd()


# In[5]:


os.chdir('C:\\Users\\Gupta Ji\\Desktop')


# In[7]:


df=pd.read_csv("diabetes.csv")


# In[8]:


df.head()


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[12]:


##scale the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[13]:


df.info()


# In[14]:


df.corr()


# In[15]:


df.isna().sum()


# In[17]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[20]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion='gini')


# In[21]:


model.fit(x_train,y_train)


# In[22]:


y_pred = model.predict(x_test)


# In[23]:


from sklearn.metrics import accuracy_score
from sklearn import tree


# In[24]:


accuracy_score(y_test,y_pred)


# In[30]:


plt.figure(figsize=(30,35))
_= tree.plot_tree(model,max_depth=5,feature_names=df.columns[:-1],class_names=['yes','no'],filled=True)


# In[ ]:




