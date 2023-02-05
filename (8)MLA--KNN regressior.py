#!/usr/bin/env python
# coding: utf-8

# In[41]:


import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[42]:


import os


# In[43]:


os.getcwd()


# In[44]:


os.chdir('C:\\Users\\Gupta Ji\\Downloads')


# In[45]:


df = pd.read_csv('Slump Test.csv')


# In[46]:


df.shape


# In[47]:


df.head()


# In[48]:


df.isnull().sum()


# In[49]:


df.dtypes


# In[50]:


#cheking for outliers
plt.figure(figsize =(15,25))
count =1
for col in df:
    plt.subplot(5,2,count)
    plt.boxplot(df[col])
    plt.title(col)
    count +=1
plt.show()    


# In[51]:


df.corr()


# In[52]:


#split X and y
X =df.iloc[:,:1].values
y =df.iloc[:,-1].values


# In[53]:


X.shape


# In[54]:


y.shape


# In[55]:


y =y.reshape(-1,1)


# In[56]:


from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
X = sc.fit_transform(X)


# In[57]:


#splitting into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state =0)


# In[58]:


#importing the model and metics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score


# In[59]:


mse =[]
r2 =[]
for i in range(1,10):
    model =KNeighborsRegressor(n_neighbors = i)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    mse_i = mean_squared_error(y_test,y_pred)
    mse.append(mse_i)
    r2_i = r2_score(y_test,y_pred)
    r2.append(r2_i)


# In[60]:


plt.figure(figsize =(8,8))
plt.plot(np.arange(1,10),mse,'r')
plt.xlabel('k-value')
plt.ylabel('mean-squared-error')
plt.title('Selecting k-value')
plt.show()


# In[61]:


model =KNeighborsRegressor(n_neighbors =2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[62]:


r2_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




