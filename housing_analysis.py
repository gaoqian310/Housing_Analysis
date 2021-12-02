#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading dataset
from sklearn.datasets import load_boston
housing = load_boston()
print(housing.keys())


# In[3]:


print (housing.DESCR)


# In[4]:


# Creating dataframe with features
housing_df = pd.DataFrame(housing.data, columns = housing.feature_names)


# In[5]:


# Adding target variable to the dataset
housing_df['MEDV'] = housing.target


# In[6]:


housing_df.head()


# In[7]:


housing_df.info()


# In[8]:


housing_df.describe()


# ## What's the price distribution of the housing?

# In[9]:


sns.set_theme(style="darkgrid")
sns.displot(housing_df['MEDV'])


# ## Correlation matrix using heatmap

# In[26]:


# Correlation matrix
housing_corr = housing_df.corr()
plt.figure (figsize=(20,12))
sns.heatmap(housing_corr,annot = True, vmin= -1 , cmap = 'YlGnBu')


# Strong negative correlation (-0.74) with % lower status of the population (LSTAT)
# 
# Strong positive correlation (0.7) with average number of rooms per dwelling (RM)

# In[ ]:




