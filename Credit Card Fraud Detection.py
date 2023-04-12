#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os


# In[5]:


os.getcwd()


# In[6]:


os.chdir("C:\\Users\\Gupta Ji\\Desktop")


# In[7]:


df=pd.read_csv("creditcard.csv")


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.shape


# In[11]:


print("shape of rows",df.shape[0])
print("shape of coulamns",df.shape[1])


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


from sklearn.preprocessing import StandardScaler 


# In[15]:


Sc=StandardScaler()
df['Amount']=Sc.fit_transform(pd.DataFrame(df['Amount']))


# In[16]:


df.head()


# In[17]:


df=df.drop(["Time"],axis=1)


# In[18]:


df.head()


# In[19]:


df.shape


# In[20]:


df.duplicated().any()


# In[21]:


df=df.drop_duplicates()


# In[22]:


df.shape


# In[23]:


284807-275663


# In[24]:


df['Class'].value_counts()


# In[25]:


import seaborn as sns


# In[26]:


sns.countplot(df['Class'])


# In[33]:


X= df.drop('Class',axis=1)
y= df['Class']


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[36]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)


# In[37]:


y_pred = log.predict(X_test)


# In[1]:


from sklearn.metrics import accuracy_score,cofusion_metrix


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:




