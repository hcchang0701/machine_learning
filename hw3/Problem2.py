#!/usr/bin/env python
# coding: utf-8

# # ML assignment 3
# ## Problem 1

# In[1]:


import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv('Concrete_Data.csv')

# rename columns
df.rename(columns=lambda x: x.split('(')[0], inplace=True)


# In[2]:


df.assign().head()


# In[3]:


df.describe(percentiles=[])


# In[4]:


df.input=df.iloc[:,0:-1]
df.input.head()


# In[5]:


df.output=df.iloc[:,-1]
df.output.head()


# In[6]:


# Show columns containing missing values
print("Columns containing missing value:", 
      df.columns[df.isna().any()].tolist())


# ## Standardization 

# In[7]:


from sklearn.preprocessing import StandardScaler
#df = StandardScaler().fit_transform(df)
#df.assign()

np_scaled = StandardScaler().fit_transform(df)
df = pd.DataFrame(np_scaled, columns=df.columns)
df.head()


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


from sklearn.metrics import mean_squared_error, r2_score
row = ['lm1', 'lm2', 'lm3', 'lm4', 'lm5', 'lm6', 'lm7', 'lm8']
col = ['MSE', 'Cor', 'R2', 'bias', 'weight']
regResult = pd.DataFrame(index=row, columns=col)


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

inputNum = 8

X, y = df.iloc[:, 0:inputNum], df.iloc[:, inputNum:inputNum+1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ## Simple linear regression
# * iteratively train linear model with each attribute

# In[14]:


axes = []
fig, axes = plt.subplots(nrows=inputNum, sharey=True, figsize=(18, 30))


for i in range(0, inputNum):
    #sns.regplot(x=df.columns[i], y=df.columns[inputNum], data=df, ax=axes[i])
    
    #df2 = pd.concat([X_train, X_test, y_train, y_test], axis=1)
    Train = pd.concat([X_train, y_train], axis=1)
    Test = pd.concat([X_test, y_test], axis=1)
    
    


    ### simple linear regression by sklearn function

    # Train linear model by training set
    reg1 = LinearRegression().fit(X_train.iloc[:, i:i+1], y_train)
    y_pred_lm = reg1.predict(X_test.iloc[:, i:i+1])
    Test['y_pred_lm'] = y_pred_lm

    # Plot outputs    
    sns.regplot(x=Test.columns[i], y='y_pred_lm', 
                data=Test, ax=axes[i], label='regression', marker='.')
    sns.scatterplot(x=Test.columns[i], y='Concrete compressive strength', 
                    data=Test, ax=axes[i], label='scatter')


    #plt.plot(X_test, y_pred_lm, color='blue', linewidth=3)

    regResult.iloc[i, 0] = mean_squared_error(y_test, y_pred_lm)
    regResult.iloc[i, 1] = reg1.score(X_train.iloc[:, i:i+1], y_train)
    regResult.iloc[i, 2] = r2_score(y_test, y_pred_lm)
    regResult.iloc[i, 3] = reg1.intercept_[0]
    regResult.iloc[i, 4] = reg1.coef_[0]
    regResult.assign()

    #plt.legend()
    #plt.show()


# In[15]:


regResult.assign()


# In[16]:


max(regResult['Cor']), max(regResult['R2'])


# In[ ]:





# ## Problem2
# * Build own gradient descent function
# * 現在只用第一個 attribute "Cement" 下去 train，可能要 train 3~5 分鐘

# ### Refresh Data

# In[17]:


df.assign().head()


# In[18]:


from sklearn.preprocessing import StandardScaler

np_scaled = StandardScaler().fit_transform(df)
df_normalized = pd.DataFrame(np_scaled, columns=df.columns)
df_normalized.head()


# In[19]:


#X, y = df.iloc[:, 0:-1], df.iloc[:, -1]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Train = pd.concat([X_train, y_train], axis=1)
Test = pd.concat([X_test, y_test], axis=1)


# In[20]:


X_train.head()


# ### Gradient Descent 

# In[25]:


# Gradient Descent 
def descent(X, y, b_current, m_current, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(X.shape[0])
    for i in range(0, X.shape[0]):
        b_gradient += -(2/N) * (y.iloc[i][0] - ((m_current * X.iloc[i][0]) + b_current))
        m_gradient += -(2/N) * X.iloc[i][0] * (y.iloc[i][0] - ((m_current * X.iloc[i][0]) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m, learning_rate * b_gradient, learning_rate * m_gradient
    #return float(new_b), float(new_m), float(learning_rate * b_gradient), float(learning_rate * m_gradient)

def gd(X, y, starting_b=0, starting_m=0, learning_rate=0.01, epochs=2000):
    b = starting_b
    m = starting_m
    step1 = 0
    step2 = 0
    stopThreshold = 0.000001
    for i in range(epochs):
        b, m, step1, step2 = descent(X, y, b, m, learning_rate)
        #print(b, m, step1, step2)
        if abs(step1) < stopThreshold or abs(step2) < stopThreshold:
            print(b, m, step1, step2)
            print("epoch: ", i)
            break
    return b, m


# In[26]:


# BB: Bias (w0)
# MM: Slope (w1)
BB, MM = gd(X_train.iloc[:, 0:1], y_train)


# In[27]:


print(BB, MM)


# ### Prediction

# In[28]:


#from sklearn.preprocessing import StandardScaler

# fit test data to our gd model
y_pred_gd = X_test.iloc[:, 0:1]*MM+BB
Test['y_pred_gd'] = X_test.iloc[:, 0:1]*MM+BB


# In[29]:


# check mean=0, std=1
print(float(y_pred_gd.mean()), float(y_pred_gd.std()))


# ### Plot the result

# In[43]:


x1 = np.linspace(-3, 3, 5000)

sns.lineplot(x1, x1*MM + BB, label='ideal gd regression line', linewidth=5)
sns.regplot(x=Test.columns[0], y='y_pred_gd', 
                data=Test, label='gd regressor pred_data', marker='.')
sns.scatterplot(x=Test.columns[0], y='Concrete compressive strength', 
                data=Test, label='test_data')


# In[32]:


# result
print(r2_score(y_test, y_pred_gd))


# In[36]:


regResult.loc['gd'] = mean_squared_error(y_test, y_pred_gd), 0, r2_score(y_test, y_pred_gd), BB, MM


# In[37]:


regResult.assign()


# In[ ]:





# ## cf. Problem1 & Problem2
# * 可看到每一項的都差不多(誤差在0.01的等級)
# * 造成兩種方法數值誤差的可能性
#     * 浮點運算誤差
#     * 我們 Gradient Descenet 的 stopThreshold 設的比較小，可能提早跳出迴圈

# In[38]:


regResult.loc[['lm1','gd']].assign()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 這下面是 sklearn 的 Stochastic Gradient Descent Regressor
# * 用全部 8 個 input attribute 下去 train，r2_score 可到 0.4

# In[ ]:


import numpy as np
from sklearn import linear_model
clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
clf.fit(X_train.iloc[:,0:8], y_train)


# In[ ]:


y_pred_sgd = clf.predict(X_test)


# In[ ]:


print('Coefficients (weight): ', clf.coef_)
print('\nIntercept (bias): ', clf.intercept_)
print('SGD Correlation: ', clf.score(X_train.iloc[:, 0:8], y_train))
print('SGD R2-score: ', r2_score(y_test, y_pred_sgd))


# In[ ]:


x1 = np.linspace(-3, 3, 5000)

plt.plot(x1, x1, label='y=x')
for i, c in enumerate(clf.coef_):
    plt.plot(x1, c*x1 + clf.intercept_, label='regression'+str(i))
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




