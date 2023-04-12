#!/usr/bin/env python
# coding: utf-8

# ### fake news detection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.chdir('C:\\Users\\Gupta Ji\\Desktop')


# In[7]:


df_fake= pd.read_csv("C:\\Users\\Gupta Ji\\Desktop\\Fake\\Fake.csv")


# In[8]:


df_true=pd.read_csv("C:\\Users\\Gupta Ji\\Desktop\\Fake\\True.csv")


# In[9]:


df_fake.head()


# In[10]:


df_true.head()


# In[14]:


df_fake.shape,df_true.shape


# In[16]:


df_marge=pd.concat([df_fake,df_true],axis=0)
df_marge.head(10)


# In[17]:


df=df_marge.drop(["subject","date","title"],axis=1)
df.head(10)


# In[18]:


df.shape


# In[33]:


df.isnull().sum()


# In[ ]:



    


# In[ ]:





# In[ ]:




