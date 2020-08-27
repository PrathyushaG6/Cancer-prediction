#!/usr/bin/env python
# coding: utf-8

# In[26]:


#importing the libraries
import pandas as pd
import matplotlib as plt
import numpy as np
import random
import warnings


# In[27]:


#importing the dataset
data = pd.read_csv(r'C:\Users\krish\anaconda3\bcsc.csv')
#data.head()
print(data)
X = data.iloc[:, [0,10]]
Y = data.iloc[:, 11]


# In[28]:


#fin missing values,if any
data.isnull()


# In[29]:


#Missing Data Percentage List
for col in data.columns:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[31]:


#data['count'].describe()


# In[32]:


num_rows = len(data.index)
low_information_cols = [] 

for col in data.columns:
    cnts = data[col].value_counts(dropna=False)
    top_pct = (cnts/num_rows).iloc[0]
    
    if top_pct > 0.95:
        low_information_cols.append(col)
        print('{0}: {1:.5f}%'.format(col, top_pct*100))
        print(cnts)
        print()


# In[34]:


#data_dedupped = data.drop('count', axis=1).drop_duplicates()
print(data.shape)
print(data_dedupped.shape)


# In[35]:


#removing duplicate rows
data.drop_duplicates(subset=None , inplace=True)


# In[36]:


print(data)


# In[37]:


data.head()


# In[44]:


data.tail()


# In[38]:


len(data)


# In[39]:


data.columns


# In[40]:


data.dtypes


# In[41]:


data.index


# In[42]:


data.age.unique()


# In[43]:


data.age.value_counts()


# In[44]:


#binning the age_categories 1 to 13 to 5 different bins "age"
bins = [1, 4, 7, 10, 13]
group_names = ['group1' , 'group2' , 'group3' , 'group4']


# In[45]:


age_categories = pd.cut(data['age'], bins, labels=group_names)
age_categories


# In[46]:


#pre processing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)


# In[47]:


#shuffling the data
data = data.sample(frac=1).reset_index(drop=True)


# In[48]:


data.head()


# In[49]:


#Splitting the dataset into the Training set and Test set to 80 and 20
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)
print (X_train.shape , Y_train.shape)
print (X_test.shape, Y_test.shape)


# In[50]:


data.tail()


# In[51]:


#ignore all caught warnings   
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
scores = []
best_svr = SVR(kernel='rbf', gamma = 'scale')
cv = KFold(n_splits=10, random_state=42, shuffle=True)
#for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
print ("Train Set                                            Test Set         ")
for train_set,test_set in cv.split(X):
    print(train_set , test_set)

    #X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]


    


# In[13]:


#from sklearn import svm
#svc = svm.SVC(kernel='linear',C=1).fit(X,Y) 


# In[ ]:


#from sklearn import cross_validation
#from sklearn.model_selection import cross_validate
#from sklearn.cross_validation import KFold, cross_val_score
#from sklearn.model_selection import KFold , cross_val_score
#Kfold = KFold(len(data),n_splits=5,shuffle=False)
#print("KfoldCrossVal score using SVM is %s" %cross_val_score(best_svr,X,Y,cv=10).mean())


# In[ ]:


#from sklearn import metrics


# In[ ]:


#sm = svc.fit(X_train,Y_train)
#Y_pred = sm.predict(X_test)
#print("Accuracy: {}%".format(sm.score(X_test, Y_test) * 100 ))


# In[52]:


#Randomforest classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42) # Instantiate model with 1000 decision trees
rf.fit(X_train, Y_train);#train model on training data


# In[53]:


predictions = rf.predict(X_test) #make predictions on test data
errors = abs(predictions - Y_test) #calculate absolute errors
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[54]:


#calculating accuracy
mape = 100 * (errors / Y_test) #calculate mean absolute percentage error mape
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[55]:


#Naive Bayes calssification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
model = GaussianNB()
model.fit(X_train,Y_train)


# In[56]:


Y_pred = model.predict(X_test)


# In[57]:


accuracy = accuracy_score(Y_test,Y_pred)*100
accuracy


# In[58]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train,Y_train)


# In[59]:


predicted = model.predict(X_test)
print(np.mean(predicted == Y_test))


# In[ ]:




