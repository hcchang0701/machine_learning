#!/usr/bin/env python
# coding: utf-8

# # ML assignment 3
# ## Problem 1

# In[1]:


import warnings; warnings.simplefilter('ignore')
import pandas as pd
df = pd.read_csv('Concrete_Data.csv')


# In[2]:


# rename columns
df.rename(columns=lambda x: x.split('(')[0], inplace=True)


# In[3]:


df.assign().head()


# In[4]:


df.describe(percentiles=[])


# In[5]:


df.input=df.iloc[:,0:-1]
df.input.head()


# In[6]:


df.output=df.iloc[:,-1]
df.output.head()


# In[7]:


# Show columns containing missing values
print("Columns containing missing value:", 
      df.columns[df.isna().any()].tolist())


# ### Visualiaztion for 9 attributes

# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[9]:


#g1 = sns.pairplot(df)
#g1.savefig("pairplot.png")


# In[10]:


# Create a figure instance, and the two subplots
inputNum = 8

axes = []
fig, axes = plt.subplots(nrows=inputNum, sharey=True, figsize=(18, 30))

for i in range(0, inputNum):
    sns.regplot(x=df.columns[i], y=df.columns[inputNum], data=df, ax=axes[i])

plt.show()


# In[11]:


# Create a figure instance, and the two subplots
inputNum = 8

axes = []
fig, axes = plt.subplots(3, 3, figsize=(24, 24))

for i in range(0, 3):
    for j in range(0, 3):
        sns.regplot(x=df.columns[i*3+j], y=df.columns[inputNum], data=df, ax=axes[i][j])

plt.show()


# ## Data Selection & Data Partition
# * For each input attribute
#     * 80% data for training
#     * 20% data for testing

# In[12]:


X, y = df.iloc[:, 0], df.iloc[:, -1]

# X = X.values.reshape(-1, 1)
# y = y.values.reshape(-1, 1)

from sklearn.metrics import mean_squared_error
row = ['lm1', 'lm2', 'lm3', 'lm4', 'lm5', 'lm6', 'lm7', 'lm8']
col = ['MSE', 'Cor(R2-score)', 'bias', 'weight']
regResult = pd.DataFrame(index=row, columns=col)


# In[13]:


from sklearn.model_selection import train_test_split
trainCol = ['X_train', 'y_train']
testCol = ['X_test', 'y_test']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[14]:


Train = pd.DataFrame(columns=trainCol)
Train.iloc[:, 0]=X_train.values
Train.iloc[:, 1]=y_train.values

Test = pd.DataFrame(columns=testCol)
Test.iloc[:, 0]=X_test.values
Test.iloc[:, 1]=y_test.values


# In[15]:


X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)


# ## Simple linear regression
# * iteratively train linear model with each attribute

# In[16]:


# simple linear regression by sklearn function
from sklearn.linear_model import LinearRegression

# Train linear model by training set
reg1 = LinearRegression().fit(X_train, y_train)
y_pred_lm = reg1.predict(X_test)
Test['y_pred_lm'] = y_pred_lm
# The coefficients
#print('Coefficients (weight): ', reg1.coef_)
#print('Intercept (bias): ', reg1.intercept_)
#print('linear model Correlation (R2-score): \n', reg1.score(X_train, y_train))


# Plot outputs
plt.scatter(X_test, y_test,  color='black', label='test data')
plt.plot(X_test, y_pred_lm, color='blue', linewidth=3, label='linear model prediction')

regResult.iloc[0, 0] = mean_squared_error(y_test, y_pred_lm)
regResult.iloc[0, 1] = reg1.score(X_train, y_train)
regResult.iloc[0, 2] = reg1.intercept_[0]
regResult.iloc[0, 3] = reg1.coef_[0]
regResult.assign()

plt.legend()
plt.show()


# In[17]:


# cf. testing data(blue) & predicted data(orange)
#sns.regplot(x=X_test.reshape(1,-1)[0], y=y_test.reshape(1,-1)[0])

#sns.scatterplot(x='X_test', y='y_test', data=Test)
#sns.lineplot(x='X_test', y='y_pred_lm', data=Test, color='orange')


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

inputNum = 8

axes = []
fig, axes = plt.subplots(nrows=inputNum, sharey=True, figsize=(18, 30))

for i in range(0, inputNum):
    #sns.regplot(x=df.columns[i], y=df.columns[inputNum], data=df, ax=axes[i])


    X, y = df.iloc[:, i], df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #df2 = pd.concat([X_train, X_test, y_train, y_test], axis=1)
    Train = pd.concat([X_train, y_train], axis=1)
    Test = pd.concat([X_test, y_test], axis=1)
    
    X_train = X_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)



    ### simple linear regression by sklearn function

    # Train linear model by training set
    reg1 = LinearRegression().fit(X_train, y_train)
    y_pred_lm = reg1.predict(X_test)
    Test['y_pred_lm'] = y_pred_lm

    # Plot outputs
    #sns.scatterplot(x=X_test.reshape(1,-1)[0], y=y_test.reshape(1,-1)[0], ax=axes[i])
    #sns.regplot(x=X_test.reshape(1,-1)[0], y=y_pred_lm.reshape(1,-1)[0], ax=axes[i])
    
    sns.regplot(x=Test.columns[0], y=Test.columns[2], 
                data=Test, ax=axes[i], label='regression', marker='.')
    sns.scatterplot(x=Test.columns[0], y=Test.columns[1], 
                    data=Test, ax=axes[i], label='scatter')


    #plt.plot(X_test, y_pred_lm, color='blue', linewidth=3)

    regResult.iloc[i, 0] = mean_squared_error(y_test, y_pred_lm)
    regResult.iloc[i, 1] = reg1.score(X_train, y_train)
    regResult.iloc[i, 2] = reg1.intercept_[0]
    regResult.iloc[i, 3] = reg1.coef_[0]
    regResult.assign()

    #plt.legend()
    #plt.show()


# In[19]:


regResult.assign()


# In[ ]:




