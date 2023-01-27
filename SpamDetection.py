#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from os import walk
from nltk.corpus import stopwords


# # Training 

# In[2]:


df= pd.read_csv('spam.csv')
df.columns


# In[3]:


X = df['EmailText'].values
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
cv = CountVectorizer() 
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)


# In[4]:


#Trying different kernels:


# In[5]:


classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))


# In[6]:


classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))


# In[7]:


classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))


# In[8]:


classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))


# # Training

# In[10]:


path = walk("enron-spam/")
Spam,Ham = [],[]

for rt,dr,files in path:
    if "spam" in str(files):
        for file in files:
            with open(rt + '/' + file,encoding='latin1') as ip:
                Spam.append(" ".join(ip.readlines()))
    if "ham" in str(files):
        for file in files:
            with open(rt + '/' + file,encoding='latin1') as ip:
                Ham.append(" ".join(ip.readlines()))

Spam = list(set(Spam))
Ham = list(set(Ham))
Data = Spam + Ham
Labels = ["spam"]*len(Spam) + ["ham"]*len(Ham)

raw_df = pd.DataFrame({
    "email":Data,
    "label":Labels
})

stopWords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words=stopWords)#,min_df=1)

email = vectorizer.fit_transform(raw_df.email.to_list())
label_encoder = sk.preprocessing.LabelEncoder()
labels = label_encoder.fit_transform(raw_df.label)

X_train,X_test,y_train,y_test = train_test_split(email,labels,train_size=0.8,random_state=42,shuffle=True)


# In[11]:


classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))


# In[12]:


classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))


# In[13]:


classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))


# In[14]:


classifier = SVC(kernel = 'sigmoid', random_state = 0)
classifier.fit(X_train, y_train)

print(classifier.score(X_test,y_test))


# # Test Folder Reading Function

# In[ ]:


def read_test_emails():
    testwalk = walk("test/")
    emails = []
    for root,dr,files in testwalk:
#         print(f"{root},{dr},{files}")
        for file in files:
            if "email"  in file:
    #             print(file)
                with open(root + file) as infile:
                    content = infile.read()
                    text = rtf_to_text(content)
                emails.append(text)
    return emails


# In[ ]:


test_vectorizer = CountVectorizer(stop_words=stopWords)#, min_df=1)

X_test_ = test_vectorizer.fit_transform(read_test_emails()).toarray()
X_test_mod = np.zeros((X_test_.shape[0],X_train.shape[1]))

# print(vectorizer.vocabulary_)

dict_train = vectorizer.vocabulary_
dict_test = test_vectorizer.vocabulary_

for key in dict_train.keys():
    if key in dict_test.keys():
        X_test_mod[:,dict_train[key]] = X_test_[:,dict_test[key]]
#         print(f"train:{dict_train[key]},test:{dict_test[key]}")
        # X_test_mod[:,dict_train[key]] = X_test_[:,dict_test[key]]X_test_mod


# In[ ]:


y_pred = classifier.predict(X_test_mod)
print(y_pred)

