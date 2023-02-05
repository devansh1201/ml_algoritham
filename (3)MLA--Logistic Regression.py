#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


import os


# In[5]:


os.getcwd()


# In[6]:


os.chdir('C:\\Users\\Gupta Ji\\Downloads')


# In[7]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[8]:


df.head()


# In[9]:


df.shape


# In[10]:


df.info()


# In[11]:


df.isna().sum()


# In[16]:


#converting categorical valus to numericl values
df = pd.get_dummies(df,drop_first = True)


# In[17]:


df.head()


# In[19]:


#checking each value counts
df['Purchased'].value_counts()


# In[20]:


#Sepreating X and y
X = df.loc[:,['Age','EstimatedSalary']].values
y = df.loc[:,['Purchased']].values


# In[21]:


y.shape


# In[22]:


#splitting in training and testing values
from sklearn.model_selection import train_test_split


# In[24]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state =0)


# In[26]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


model = LogisticRegression()


# In[29]:


model.fit(X_train,y_train)


# In[30]:


#predictions
y_pred = model.predict(X_test)


# In[32]:


from sklearn.metrics import accuracy_score,confusion_matrix
cfm = confusion_matrix(y_test,y_pred)
cfm


# In[33]:


accuracy_score(y_test,y_pred)


# In[ ]:




