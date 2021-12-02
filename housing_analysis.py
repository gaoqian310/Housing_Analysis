#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Importing libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


# Loading dataset
from sklearn.datasets import load_boston
housing = load_boston()
print(housing.keys())


# In[17]:


print (housing.DESCR)


# In[19]:


# Creating dataframe with features
housing_df = pd.DataFrame(housing.data, columns = housing.feature_names)


# In[20]:


# Adding target variable to the dataset
housing_df['MEDV'] = housing.target


# In[21]:


housing_df.head()


# In[22]:


housing_df.info()


# In[23]:


housing_df.describe()


# In[45]:


sns.set_theme(style="darkgrid")
sns.displot(housing_df['MEDV'])


# In[ ]:





# In[ ]:




