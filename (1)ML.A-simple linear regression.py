#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


#importing the datasets
dataset = pd.read_csv('Salary_Data.csv')


# In[6]:


import os


# In[7]:


os.getcwd()


# In[8]:


os.chdir('C:\\Users\\Gupta Ji\\Downloads')


# In[9]:


dataset = pd.read_csv('Salary_Data.csv')


# In[ ]:


dataset.shape


# In[27]:


#seperating X and Y
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[28]:


# divinding into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state = 0)


# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


regressor = LinearRegression()


# In[31]:


regressor.fit(X_train,y_train)


# In[32]:


#prediction
y_pred = regressor.predict(X_test)


# In[33]:


#visualising the trainig set
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color ='blue')
plt.title('Salary vs Year of experiece(training set)')
plt.xlabel('YearsExperience')
plt.ylabel('salary')
plt.show()


# In[34]:


dataset.head()


# In[35]:


#visualising the training set
plt.scatter(X_test,y_test,color ='red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Salary vs year of experience(testing set)')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()


# In[36]:


#checking the score
from sklearn.metrics import r2_score


# In[37]:


r2_score(y_test,y_pred)


# In[ ]:




