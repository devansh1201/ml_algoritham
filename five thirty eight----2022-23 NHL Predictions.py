#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


import os


# In[4]:


os.getcwd()


# In[5]:


os.chdir('C:\\Users\\Gupta Ji\\Desktop\\nhl-forecasts')


# In[6]:


df=pd.read_csv("nhl_elo.csv")


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.shape


# In[10]:


df.describe()


# In[13]:


df[home_team_sore].value_counts().plot(kind ='barh',figsize =(8,6))
plt.xlabel("count",labelpad =14)
plt.ylabel("Target Variable",labelpad = 14)
plt.title("count of TARGET home_team_score ", y=1.02);


# In[14]:


df['away_team_score'].value_counts().plot(kind ='barh',figsize =(8,6))
plt.xlabel("count",labelpad =14)
plt.ylabel("Target Variable",labelpad = 14)
plt.title("count of TARGET away_team_score", y=1.02);


# In[22]:


#droping some collmuns which have no singnificance
df.drop(['playoff'],axis = 1,inplace = True)


# In[23]:


#droping some collmuns which have no singnificance
df.drop(['neutral'],axis = 1,inplace = True)


# In[24]:


df.head()


# In[26]:


df.shape


# In[ ]:




