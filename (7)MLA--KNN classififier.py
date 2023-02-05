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


df = pd.read_csv('Social_Network_Ads (1).csv')


# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df.isna().sum()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


#droping user id column as it is not an important feature
df.drop('User ID',axis = 1,inplace = True )


# In[12]:


df.head()


# In[13]:


#converting categorical values to numerical 
df = pd.get_dummies(df,drop_first = True)


# In[14]:


df.head()


# In[15]:


#seperating X and y


# In[16]:


X = df.loc[:,['Age','EstimatedSalary','Gender_Male']].values
y =df.loc[:,['Purchased']].values


# In[17]:


X


# In[18]:


#y


# In[19]:


from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X = sc.fit_transform(X)


# In[20]:


X


# In[21]:


#model was expecting a 1d array as input
y = y.reshape(-1)


# In[22]:


y.shape


# In[27]:


#splitting the data into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0 ,test_size = 0.25)


# In[28]:


#importing classification metrics
from sklearn.metrics import confusion_matrix,accuracy_score


# In[29]:


from sklearn.neighbors import KNeighborsClassifier


# In[35]:


acc_list =[]
err_list =[]
for i in range(1,25):
    model = KNeighborsClassifier(n_neighbors =i)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    
#   print("For k ={},accuracy ={}".format(i,acc))
    acc_list.append(acc)
    err_list.append(1-acc)


# In[36]:


plt.plot(list(range(1,25)),err_list,c ='r')
plt.title('Error rate v/s K')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()


# In[37]:


model =KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[38]:


confusion_matrix(y_test,y_pred)


# In[39]:


accuracy_score(y_test,y_pred)


# In[ ]:




