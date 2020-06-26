import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the data
df = pd.read_csv('news.csv')  #dataset news
# Get shape and head
print(df.shape)
#print(df.head())
#Get the labels
labels = df.label
#print(labels.head())  #printing first 5 labels


#splitting into training and testing data set
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)   #splitting on basis of text and labels,test data=20% (0.2) ,
#  random_state is given a constant number to get same training and testing data every time we execute program



#  Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
#  Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
# Predict on the test set and calculate accuracy

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#for train data
 # x_pred=pac.predict(tfidf_train)
 # score2=accuracy_score(x_train,x_pred)
 # print(f'Accuracy for train data: {round(score2*100,2)}%')

#Build confusion matrix
print(confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']))    #o/p: [590 48
                                                                       # 43 586 ]  , it means we have 590 true positives,586 true negatives
                                                                       #43 false positive & 48 false negatives
#confusion matrix is used on test data

X =  tfidf_vectorizer.transform(['Kerry to go to Paris in gesture of sympathy']) #to ask for any random news ,first need to transform it
# X = X.reshape(-1,1)
Y = pac.predict(X)  #predicting val for this news

print(Y) # printing fake or real
