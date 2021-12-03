#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Loading dataset
from sklearn.datasets import load_boston
housing = load_boston()
print(housing.keys())


# In[4]:


print (housing.DESCR)


# In[5]:


# Creating dataframe with features
housing_df = pd.DataFrame(housing.data, columns = housing.feature_names)


# In[6]:


# Adding target variable to the dataset
housing_df['MEDV'] = housing.target


# In[7]:


housing_df.head()


# In[8]:


housing_df.info()


# In[9]:


housing_df.describe()


# ## What's the price distribution of the housing?

# In[10]:


sns.set_theme(style="darkgrid")
plt.figure (figsize=(10,6))
sns.distplot(housing_df['MEDV'], axlabel = 'Median value of owner-occupied homes in $1000')


# <span style="color:red">FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).</span>

# ## Reproduce distplot with kdeplot and histplot

# In[11]:


sns.set_theme(style="whitegrid")
plt.figure (figsize=(15,8))

ax = sns.kdeplot(data=housing_df['MEDV'])
ax.set(ylabel='density', xlabel='Median value of owner-occupied homes in $1000')

ax2 = ax.twinx()
ax2 = sns.histplot(housing_df['MEDV'], ax=ax2)
ax2.set_ylabel('count')

plt.tight_layout()
plt.show()


# ## Correlation matrix using heatmap

# In[16]:


# Correlation matrix
housing_corr = housing_df.corr()
plt.figure (figsize=(20,12))
sns.heatmap(housing_corr,annot = True, vmin= -1 , cmap = 'YlGnBu')
plt.show()


# #### Conclusion
# Strong negative correlation (-0.74) with % lower status of the population (LSTAT)
# 
# Strong positive correlation (0.7) with average number of rooms per dwelling (RM)

# In[21]:


# Jointplots for high correlations - lower status population
plt.figure (figsize=(10,10))
sns.jointplot(x = 'LSTAT', y = 'MEDV', data = housing_df, kind = 'reg', height = 10, color = 'green')
plt.show()


# #### Conclusion
# House pricing is negitave corelated to the lower status ofthe population. likely dependent on the lower status of the population.
# - LSTAT    % lower status of the population
# - MEDV     Median value of owner-occupied homes in $1000's

# In[26]:


# Jointplots for high correlations - number of rooms
plt.figure (figsize=(10,10))
sns.jointplot(x = 'RM', y = 'MEDV', data = housing_df, kind = 'hex', color = 'teal', height = 10)
plt.show()


# #### Conclusion
# House pricing is postive corelated to the number of rooms per dwelling. 
# 
# Lots of houses with 6 rooms are at a price around $20k.
# 
# - RM       average number of rooms per dwelling
# - MEDV     Median value of owner-occupied homes in $1000's

# ## Creating a Machine Learning Model

# - Prepare the dataset
# - Split the dataset into training (75%) and test (25%) sets

# In[31]:


# Preparing the dataset
X = housing_df.drop(['MEDV'], axis = 1)
Y = housing_df['MEDV']


# In[32]:


# Splitting into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=100)


# ### First Model: Linear Regression

# In[33]:


# Training the Linear Regression model
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[34]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


# ##### Using Root-mean-square error (RMSE) and R squared (r2_score) to evaluate the model

# In[35]:


# Evaluating the Linear Regression model for the test set
from sklearn.metrics import mean_squared_error, r2_score
predictions = lm.predict(X_test)
RMSE_lm = np.sqrt(mean_squared_error(Y_test, predictions))
r2_lm = r2_score(Y_test, predictions)

print('RMSE_lm = {}'.format(RMSE_lm))
print('R2_lm = {}'.format(r2_lm))


# #### Conclusion
# 
# This model gives us an RMSE of about 5.2. 
# 
# R squared value of 0.72 means that this linear model explains 72% of the total response variable variation.
