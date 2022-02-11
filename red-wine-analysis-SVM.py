#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import svm


# ### Clean the Data 

# In[6]:


redWine = pd.read_csv('/Users/vincenzo/Desktop/Repos/tts-ds-fundamentals-course-main/datascience/python/tts-datascience-ml-wine-analysis/wine_red_white/winequality-red.csv', sep = ';')
redWine.head()


# In[14]:


#Cleaning the data
redWine.drop_duplicates(inplace=True)
redWine.drop_duplicates(inplace=True)
#Checks to male sure no leftover non applicable terms in the data sets
print('NAs in Red Wine:', redWine.isna().any().any(),'\n')


# ### Red Wine Characteristics Correlation

# In[20]:


plt.figure(figsize = (16,6))
heatmap = sns.heatmap(redWine.corr(), vmin = -1, vmax = 1, annot = True, cmap = 'BrBG')
heatmap.set_title('Correlation Heatmap', fontdict = {'fontsize': 18}, pad = 12);
plt.savefig('heatmap.png', dpi = 300, bbox_inches = 'tight')


# In[28]:


fig, axes = plt.subplots(1)
sns.regplot(x = 'quality', y = 'alcohol', data = redWine, x_estimator = np.mean)
fig.set_figheight(8)
fig.set_figwidth(12)
plt.tight_layout(w_pad = 10)


# In[31]:


# Correlation with output variable
cor = redWine.corr()
cor_target = abs(cor['quality'])

# Select highly correlated characteristics
relevant_characteristics = cor_target[cor_target > 0.45]
relevant_characteristics


# In[34]:


rating_scale = ['bad', 'good', 'great']
categories = pd.cut(redWine['quality'], bins = [2, 4, 6, 8], labels = rating_scale)
redWine['quality'] = categories


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[40]:


x = redWine.loc[:, ['alcohol']]
y = redWine.loc[:, ['quality']]


# In[41]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[42]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[43]:


# Standardize the data
from sklearn.preprocessing import StandardScaler 
SC = StandardScaler()
x_train = SC.fit_transform(x_train)
x_test = SC.fit_transform(x_test)


# In[44]:


from sklearn.svm import SVC
CL = SVC(kernel = 'poly', degree = 2)
CL.fit(x_train, y_train)
y_pred = CL.predict(x_test)


# In[53]:


from sklearn.metrics import confusion_matrix
ax = plt.subplot()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = '2.0f')
ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
ax.set_title('Red Wine Confusion Matrix')
ax.xaxis.set_ticklabels(['bad', 'good', 'great']); ax.yaxis.set_ticklabels(['bad', 'good', 'great'])


# In[55]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = CL, X = x_train,
                            y = y_train, cv = 10)
accuracies.mean()


# In[ ]:




