#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


# reading the data

train = pd.read_csv('HR-Employee-Attrition.csv')


# In[3]:


import os


# In[4]:


os.getcwd()


# In[5]:


os.chdir('C:\\Users\\Gupta Ji\\Downloads')


# In[6]:


os.getcwd()


# In[8]:


train = pd.read_csv('HR-Employee-Attrition.csv')
test = pd.read_csv('HR-Employee-Attrition.csv')

#getting their shapes
print("Shape of train:",train.shape)
print("Shape of train:",train.shape)


# In[9]:


train.head()


# In[10]:


test.head()


# In[11]:


#describing the training set
train.describe(include ='all')


# In[12]:


train.info()


# In[13]:


# checking if there is any NULL value in the dataset

train.isnull().any()


# In[14]:


test.isnull().sum()


# In[15]:


# looking at the most popular departments
from wordcloud import WordCloud
from wordcloud import STOPWORDS

stopword = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'white',stopwords = stopword).generate(str(train['Department']))

plt.rcParams['figure.figsize'] = (12,8)
print(wordcloud)
plt.imshow(wordcloud)
plt.title('Most Popular Departments',fontsize = 30)
plt.axis('off')
plt.show()


# In[16]:


#checking the no. of Employees()
train['EmployeeCount'].value_counts()


# In[17]:


# plotting a scatter plot

plt.hist(train['EmployeeCount'])
plt.title('plot  to show the gap in EmployeeCount and non-EmployeeCount',fontsize = 30)
#plt.xlabel('0 -No Promotion and 1 - Promotion ', fontsize = 20)
plt.ylabel('count')
plt.show()


# In[20]:


# cheking the distribution of the TotalWorkingYears score of the Employees

plt.rcParams['figure.figsize'] = (15,7)
sns.distplot(train['TotalWorkingYears'],color = 'yellow')
plt.title('Distribution of Training Score among the Employees',fontsize = 30)
plt.xlabel('Average Working Score',fontsize = 20)
plt.ylabel('count')
plt.show()


# In[21]:


train['Education'].value_counts()


# In[ ]:




