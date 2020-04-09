#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#importing the dataset
df = pd.read_csv('BC.csv')

#Check the info 
df.info()

#Remove the unnamed column
df = df.drop(['Unnamed: 32'],axis =1 )
#Remove the Id column
df = df.drop(['id'],axis =1)

#Checking the names of all columns
df.columns

#Dividing the columns into different groups
features_mean = list(df.columns[1:11])
features_se = list(df.columns[11:21])
features_worst = list(df.columns[21:])

#Dependent variable
y = df.iloc[:,0].values
#Independent variable
X = df.iloc[:,1:].values

#encoding the dependent variable as 0s and 1s for ease of access
from sklearn.preprocessing import LabelEncoder as LE
le = LE()
y = le.fit_transform(y)

#Check for usefull features
#1.We will check for the columns ending with "mean" in this section
corr = df[features_mean].corr()
#Usefull features
prediction_var = ['texture_mean','perimeter_mean',
                  'smoothness_mean',
                  'compactness_mean','symmetry_mean']

#Splitting into train,test sets
from sklearn.model_selection import train_test_split as tts
train , test = tts(df , test_size = 0.3)
train_X = train[prediction_var]
train_y = train.diagnosis
test_X = test[prediction_var]
test_y = test.diagnosis

#Create your SVM model here.We are using the Gaussian RBF kernel
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(train_X,train_y)
pred_y = classifier.predict(test_X)

#Check for the correct and incorrect predictions
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y,pred_y)

#Check for the Accuracy of the model
sc0=classifier.score(test_X,test_y)



#Check for useless features
#2.We will check our for columns ending with "worst" in this section
corr1 = df[features_worst].corr()
#Usefull features
prediction_var1 = ['texture_worst','perimeter_worst',
                  'smoothness_worst',
                  'compactness_worst','symmetry_worst']

#Splitting into train,test sets
from sklearn.model_selection import train_test_split as tts
train , test = tts(df , test_size = 0.3)
train_X_1 = train[prediction_var1]
train_y_1 = train.diagnosis
test_X_1 = test[prediction_var1]
test_y_1 = test.diagnosis

#Make your SVM Model here
from sklearn.svm import SVC
classifier_1 = SVC()
classifier_1.fit(train_X_1,train_y_1)

#predicted list
pred_y_1 = classifier_1.predict(test_X_1)

#Check for correct and incorrect predictions 
from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(test_y_1,pred_y_1)

#check for accuracy of your model
sc1=classifier_1.score(test_X_1,test_y_1)



#Check for useless features
#3.We will check our for columns ending with "se" in this section
corr2 = df[features_se].corr()
#Usefull features
prediction_var2 = ['texture_se','perimeter_se',
                  'smoothness_se',
                  'compactness_se','symmetry_se']

#Splitting into train,test sets
from sklearn.model_selection import train_test_split as tts
train , test = tts(df , test_size = 0.3)
train_X_2 = train[prediction_var2]
train_y_2 = train.diagnosis
test_X_2 = test[prediction_var2]
test_y_2 = test.diagnosis

#Make your SVM Model here
from sklearn.svm import SVC
classifier_2 = SVC()
classifier_2.fit(train_X_2,train_y_2)

#predicted list
pred_y_2 = classifier_2.predict(test_X_2)

#Check for correct and incorrect predictions 
from sklearn.metrics import confusion_matrix
cm_2 = confusion_matrix(test_y_2,pred_y_2)

#check for accuracy of your model
sc2=classifier_2.score(test_X_2,test_y_2)

#Maximum accuracy of the model among the top 3
print(max(sc0,sc1,sc2))



