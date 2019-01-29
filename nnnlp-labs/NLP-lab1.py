import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import nltk


'''
data = np.array([[1,2],[2,3],[3,4],[4,5],[5,6]])
x = data[:,0]
y= data[:,1]
plt.scatter(x,y)
plt.grid(True)
plt.show()
'''
'''
vectorizer = CountVectorizer(min_df=1)
content = ["How to format my hard disk", "Hard disk format problems"]
X = vectorizer.fit_transform(content)
featureNames = vectorizer.get_feature_names()
print (featureNames)
print(X.toarray())
'''

'''
categories = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories,shuffle=True,random_state=42)
vectorizer = CountVectorizer()
train_counts = vectorizer.fit_transform(twenty_train.data)
print('Frequency of the word "Algorithm":')
print(vectorizer.vocabulary_.get('algorithm'))
print('Number of terms extracted: ')
print(len(vectorizer.get_feature_names()))
'''
