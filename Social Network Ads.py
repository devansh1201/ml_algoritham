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


os.chdir('C:\\Users\\Gupta Ji\\Desktop')


# In[5]:


df = pd.read_csv("Social_Network_Ads.csv")


# In[6]:


df.head()


# In[7]:


df.drop('User ID',axis=1,inplace=True)


# In[8]:


df.head()


# In[9]:


df = pd.get_dummies(df,drop_first=True)


# In[10]:


df.head()


# In[11]:


df.describe()


# In[12]:


x=df.loc[:,['Age','EstimatedSalary','Gender_Male']].values
y=df.loc[:,['Purchased']].values


# In[13]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(x)


# In[14]:


X


# In[15]:


y


# In[16]:


y.shape


# In[17]:


Y=y.reshape(-1)


# In[18]:


Y.shape


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)


# In[22]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score


# In[24]:


acc_list =[]
err_list=[]
for i in range(1,25):
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    acc_list.append(acc)
    err_list.append(1-acc)
    


# In[25]:


acc_list


# In[27]:


plt.plot(list(range(1,25)),err_list,c='r')
plt.title('Error rate v/s K')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()


# In[32]:


model =KNeighborsClassifier(n_neighbors=5)


# 

# In[33]:


model.fit(x_train,y_train)


# In[34]:


y_pred =model.predict(x_test)


# In[38]:


confusion_matrix(y_test,y_pred)


# In[40]:


accuracy_score(y_test,y_pred)


# In[ ]:




