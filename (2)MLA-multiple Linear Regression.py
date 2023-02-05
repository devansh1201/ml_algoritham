#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[9]:


import os


# In[10]:


os.getcwd()


# In[11]:


os.chdir('C:\\Users\\Gupta Ji\\Downloads')


# In[51]:


df = pd.read_csv('Carseats.csv')


# In[52]:


df.shape


# In[53]:


df.head()


# In[54]:


#dropping Unnamed:0 column
#df.drop(df.columns[0],axis = 1,inplace = True)


# In[55]:


df.head()


# In[56]:


#check for missing values
df.isnull().sum()


# In[64]:


#check for categorical variables
df.dtypes


# In[96]:


#cheking the unique values in each categorical columns
#also checking for outliers in non categorical columns
plt.figure(figsize = (15,15))
count = 1
for col in df:
    if(df[col].dtype =='0'):
        print("Unique values in {} = {}".format(col,df[col].unique()))
    else:
        plt.subplot(5,2,count)
        plt.boxplot(df[col])
        plt.title(col)
        count +=1
        
plt.show()


# In[59]:


df.corr()


# In[60]:


df2 = df.copy()
df2['ShelveLoc'].head()


# In[62]:


#applying label encoding on Shelveloc column
my_dict = {
    'Bad':0,
    'Medium':1,
    'Good':2
}
df2['ShelveLoc_new'] = df2['ShelveLoc'].map(my_dict)
df2['ShelveLoc_new'].head()


# In[94]:


#droping old column
df2.drop('ShelveLoc',axis =1,inplace = True)


# In[67]:


#one hot encoding the other two columns and droping the first two rows
df2 = pd.get_dummies(df2,drop_first =True)


# In[68]:


df2.corr()


# In[69]:


df2.dtypes


# In[84]:


#split X and y
X = df2.iloc[:,1:].values
y = df2.iloc[:,0].values


# In[85]:


X.shape


# In[86]:


y.shape


# In[87]:


y = y.reshape(-1,1)


# In[105]:


#splitting into train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state = 0)


# In[106]:


#creating linear regession model
from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[107]:


#fitting the training data
model.fit(X_train,y_train)


# In[108]:


#predictions
y_pred= model.predict(X_test)


# In[110]:


#cheking the r2score
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:




