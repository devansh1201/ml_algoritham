#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd 
import numpy as np
import datetime
from time import strftime 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# In[10]:


#reading the dataset
base_data = pd.read_csv('household.csv')


# In[11]:


import os


# In[12]:


os.getcwd()


# In[13]:


os.chdir('C:\\Users\\Gupta Ji\\Downloads')


# In[14]:


os.getcwd()


# In[15]:


base_data = pd.read_csv('household.csv')


# In[16]:


base_data


# In[17]:


base_data.shape


# In[18]:


base_data.info()


# In[19]:


base_data.head(5)


# In[20]:


#changing the name of some cloumns
base_data = base_data.rename(columns = {'hello123':'hello123'})


# In[21]:


base_data.columns


# In[22]:


base_data.info()


# In[23]:


#droping some collmuns which have no singnificance
base_data.drop(['nzhec'],axis = 1,inplace = True)


# In[24]:


base_data


# In[25]:


base_data.info()


# In[26]:


base_data.describe()


# In[27]:


base_data['level'].value_counts().plot(kind ='barh',figsize =(8,6))
plt.xlabel("count",labelpad =14)
plt.ylabel("Target Variable",labelpad = 14)
plt.title("count of TARGET per category", y=1.02);


# In[29]:


# calculating the % of appointments or not
100*base_data['level'].value_counts()


# In[30]:


base_data['level'].value_counts()


# In[ ]:




