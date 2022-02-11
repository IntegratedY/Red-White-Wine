#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import svm


# In[2]:


whiteWine = pd.read_csv('/Users/vincenzo/Desktop/Repos/tts-ds-fundamentals-course-main/datascience/python/tts-datascience-ml-wine-analysis/wine_red_white/winequality-red.csv', sep = ';')
whiteWine.head()


# In[4]:


#Cleaning the data
whiteWine.drop_duplicates(inplace=True)
whiteWine.drop_duplicates(inplace=True)
#Checks to male sure no leftover non applicable terms in the data sets
print('NAs in White Wine:', whiteWine.isna().any().any(),'\n')


# ### White Wine Characteristics Correlation

# In[6]:


plt.figure(figsize = (16,6))
heatmap = sns.heatmap(whiteWine.corr(), vmin = -1, vmax = 1, annot = True, cmap = 'BrBG')
heatmap.set_title('Correlation Heatmap', fontdict = {'fontsize': 18}, pad = 12);
plt.savefig('heatmap.png', dpi = 300, bbox_inches = 'tight')


# In[7]:


fig, axes = plt.subplots(1)
sns.regplot(x = 'quality', y = 'alcohol', data = whiteWine, x_estimator = np.mean)
fig.set_figheight(8)
fig.set_figwidth(12)
plt.tight_layout(w_pad = 10)


# In[8]:


# Correlation with output variable
cor = whiteWine.corr()
cor_target = abs(cor['quality'])

# Select highly correlated characteristics
relevant_characteristics = cor_target[cor_target > 0.45]
relevant_characteristics


# In[9]:


rating_scale = ['bad', 'good', 'great']
categories = pd.cut(whiteWine['quality'], bins = [2, 4, 6, 8], labels = rating_scale)
whiteWine['quality'] = categories


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[11]:


x = whiteWine.loc[:, ['alcohol']]
y = whiteWine.loc[:, ['quality']]


# In[15]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(y)


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[17]:


# Standardize the data
from sklearn.preprocessing import StandardScaler 
SC = StandardScaler()
x_train = SC.fit_transform(x_train)
x_test = SC.fit_transform(x_test)


# In[18]:


from sklearn.svm import SVC
CL = SVC(kernel = 'poly', degree = 2)
CL.fit(x_train, y_train)
y_pred = CL.predict(x_test)


# In[19]:


from sklearn.metrics import confusion_matrix
ax = plt.subplot()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = '2.0f')
ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels')
ax.set_title('White Wine Confusion Matrix')
ax.xaxis.set_ticklabels(['bad', 'good', 'great']); ax.yaxis.set_ticklabels(['bad', 'good', 'great'])


# In[20]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = CL, X = x_train,
                            y = y_train, cv = 10)
accuracies.mean()


# In[ ]:




