#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
data=pd.read_csv('apples_and_oranges.csv')


# In[5]:


data


# In[6]:



data.info()


# In[7]:


data.head()


# In[8]:


#Splitting the dataset into training and test samples

from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(data, test_size = 0.2, random_state = 1)


# In[9]:


#Classifying the predictors and target

X_train = training_set.iloc[:,0:2].values
Y_train = training_set.iloc[:,2].values
X_test = test_set.iloc[:,0:2].values
Y_test = test_set.iloc[:,2].values


# In[22]:


#Initializing Support Vector Machine and fitting the training data

from sklearn.svm import SVC
classifier = SVC(kernel='rbf',gamma=0.8, random_state = 1)
classifier.fit(X_train,Y_train)


# In[11]:


#Predicting the classes for test set

Y_pred = classifier.predict(X_test)


# In[12]:


#Attaching the predictions to test set for comparing

test_set["Predictions"] = Y_pred
test_set


# In[13]:


#8 predictions have gone wrong
from sklearn.metrics import confusion_matrix
from sklearn import metrics
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
print(metrics.classification_report(Y_test,Y_pred))
accuracy = float(cm.diagonal().sum())/len(Y_test)


# In[14]:


#calculate the accuracy using the confusion matrix as follows :

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)


# In[15]:


#Visualizing the classifier
#Before we visualize we might need to encode
#We can achieve that using the label encoder.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)


# In[16]:


#After encoding , fit the encoded data to the SVM

from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state = 1)
classifier.fit(X_train,Y_train)


# In[17]:


Y_pred = classifier.predict(X_test)


# In[18]:


test_set["Predictions"] = Y_pred
test_set


# In[19]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# In[20]:


#visualizing the classifier

plt.figure(figsize = (7,7))
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('black', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'orange'))(i), label = j)
plt.title('Fruits')
plt.xlabel('Weight In Grams')
plt.ylabel('Size in cm')
plt.legend()
plt.show()


# In[21]:


#visualizing the predictions
plt.figure(figsize = (7,7))
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('black', 'white')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('red', 'orange'))(i), label = j)
plt.title('Fruits pedictions ')
plt.xlabel('Weight In Grams')
plt.ylabel('Size in cm')
plt.legend()
plt.show()


# In[ ]:




