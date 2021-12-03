#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Importing libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


# Loading dataset
from sklearn.datasets import load_boston
housing = load_boston()
print(housing.keys())


# In[38]:


print (housing.DESCR)


# In[39]:


# Creating dataframe with features
housing_df = pd.DataFrame(housing.data, columns = housing.feature_names)


# In[40]:


# Adding target variable to the dataset
housing_df['MEDV'] = housing.target


# In[41]:


housing_df.head()


# In[42]:


housing_df.info()


# In[43]:


housing_df.describe()


# ## What's the price distribution of the housing?

# In[44]:


sns.set_theme(style="darkgrid")
plt.figure (figsize=(10,6))
sns.distplot(housing_df['MEDV'], axlabel = 'Median value of owner-occupied homes in $1000')


# <span style="color:red">FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).</span>

# ## Reproduce distplot with kdeplot and histplot

# In[45]:


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

# In[46]:


# Correlation matrix
housing_corr = housing_df.corr()
plt.figure (figsize=(20,12))
sns.heatmap(housing_corr,annot = True, vmin= -1 , cmap = 'YlGnBu')
plt.show()


# #### Conclusion
# Strong negative correlation (-0.74) with % lower status of the population (LSTAT)
# 
# Strong positive correlation (0.7) with average number of rooms per dwelling (RM)

# In[47]:


# Jointplots for high correlations - lower status population
plt.figure (figsize=(10,10))
sns.jointplot(x = 'LSTAT', y = 'MEDV', data = housing_df, kind = 'reg', height = 10, color = 'green')
plt.show()


# #### Conclusion
# House pricing is negitave corelated to the lower status ofthe population. likely dependent on the lower status of the population.
# - LSTAT    % lower status of the population
# - MEDV     Median value of owner-occupied homes in $1000's

# In[48]:


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

# In[49]:


# Preparing the dataset
X = housing_df.drop(['MEDV'], axis = 1)
Y = housing_df['MEDV']


# In[50]:


# Splitting into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=100)


# ### First Model: Linear Regression

# In[51]:


# Training the Linear Regression model
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[52]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)


# ###### Using Root-mean-square error (RMSE) and R squared (r2_score) to evaluate the model

# In[53]:


# Evaluating the Linear Regression model for the test set
from sklearn.metrics import mean_squared_error, r2_score
predictions = lm.predict(X_test)
RMSE_lm = np.sqrt(mean_squared_error(Y_test, predictions))
r2_lm = r2_score(Y_test, predictions)

print('RMSE_lm = {}'.format(RMSE_lm))
print('R2_lm = {}'.format(r2_lm))


# ###### Conclusion
# 
# This model gives us an RMSE of about 5.2. 
# 
# R squared value of 0.72 means that this linear model explains 72% of the total response variable variation.

# ### Second Model: Random Forest

# In[54]:


# Training the Random Forest model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 10, random_state = 100)
rf.fit(X_train, Y_train)


# In[55]:


RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
max_features='auto', max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
oob_score=False, random_state=100, verbose=0, warm_start=False)


# In[56]:


# Evaluating the Random Forest model for the test set
predictions_rf = rf.predict(X_test)
RMSE_rf = np.sqrt(mean_squared_error(Y_test, predictions_rf))
r2_rf = r2_score(Y_test, predictions_rf)

print('RMSE_rf = {}'.format(RMSE_rf))
print('R2_rf = {}'.format(r2_rf))


# In[ ]:




