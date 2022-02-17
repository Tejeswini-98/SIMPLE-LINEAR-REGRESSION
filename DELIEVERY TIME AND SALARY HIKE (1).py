#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plot
import seaborn as sns
import statsmodels.formula.api as smf
 
import warnings
warnings.filterwarnings('ignore')


# In[6]:


delivery_time = pd.read_csv("delivery_time.csv")
delivery_time


# ### EDA and Data Visualization

# In[7]:


delivery_time.info()


# In[14]:


sns.distplot(delivery_time['Delivery Time'])
plot.title('Normality test for Delivery time')
plot.show()


# In[15]:


sns.distplot(delivery_time['Sorting Time'])
plot.title("Normality test for Sorting time")
plot.show()


# In[17]:


delivery_time=delivery_time.rename({'Delivery Time':'d_t','Sorting Time':'s_t'},axis=1)


# In[18]:


delivery_time.corr()


# ### Model Buliding

# In[19]:


model =smf.ols('d_t~s_t',data=delivery_time).fit()
model


# In[9]:


sns.regplot(x="s_t",y="d_t",data=delivery_time)


# ### Model Training

# In[10]:


model.params


# In[20]:


(model.tvalues,model.pvalues)


# In[12]:


model.rsquared ,model.rsquared_adj


# In[13]:


delivery_time=(6.582734) + (1.649020) *(5)
delivery_time


# ### Model Predictions

# ### Manual prediction for say sorting time 5

# In[14]:


new_data=pd.Series([5,8])
new_data


# In[15]:


data_pred=pd.DataFrame(new_data,columns=['s_t'])


# In[16]:


data_pred


# In[17]:


model.predict (data_pred)


# ### =========================================================================================
# 

# ### 2) Salary_hike -> Build a prediction model for Salary_hike
# 

# ### Import Data

# In[21]:


salary=pd.read_csv("Salary_Data (1).csv")


# In[22]:


salary


# In[27]:


salary.shape


# In[28]:


salary.describe()


# In[29]:


salary.isnull().sum()


# ### Data Visualization

# In[31]:


sns.distplot(salary['YearsExperience'])
plot.title("YearsExperience")
plot.show()


# In[32]:


sns.distplot(salary['Salary'])
plot.title(" Normality test for Salary")
plot.show()


# In[33]:


salary.corr()


# In[35]:


model=smf.ols('Salary~YearsExperience',data=salary).fit()
model


# In[23]:


sns.regplot(x='YearsExperience',y='Salary',data=salary)


# In[36]:


model.params


# In[38]:


(model.tvalues,model.pvalues)


# In[39]:


model.rsquared,model.rsquared,model.rsquared_adj


# In[40]:


salary_hike=(25792.200199)+(9449.962321)*(3)
salary_hike


# In[28]:


data=pd.Series([3,5])


# In[29]:


data


# In[30]:


data_pred=pd.DataFrame(data,columns=["YearsExperience"])


# In[31]:


data_pred


# In[32]:


model.predict(data_pred)


# In[ ]:




