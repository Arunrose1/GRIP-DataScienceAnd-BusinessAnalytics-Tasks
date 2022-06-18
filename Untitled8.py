#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


d="http://bit.ly/w-data"
data=pd.read_csv(d)


# In[3]:


data.describe()


# In[4]:


data.head()


# In[8]:


sns.set(rc={'figure.figsize':(7,5)})
sns.distplot(data.Scores,bins=30)
plt.show()


# In[10]:


x=data['Hours'].values
y=data['Scores'].values
print(x)
print(y)


# In[16]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,)
print(y_test)


# In[29]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
X_train=np.reshape(X_train, (20,1))
y_train=np.reshape(y_train, (20,1))
X_test=np.reshape(X_test, (5,1))
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)


# In[32]:


l=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.show()


# In[33]:


time=9.25
time=np.reshape(time,(1,1))
predic=regressor.predict(time)
print("Given time=",time)
print("Predicted score:",predic)


# In[34]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




