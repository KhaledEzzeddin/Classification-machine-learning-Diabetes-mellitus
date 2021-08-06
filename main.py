 #_________________libraries__________
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
#__________________read the file____________
df=pd.read_csv("diabetes.csv")
df.head()
#__________________preparing the database to training the Algorithme ____________
columns=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
         "BMI","DiabetesPedigreeFunction","Age","Outcome"]
labels=df["Outcome"].values
features=df[list(columns)].values
print("\nthe labels is\n",labels)
print("\nthe features is\n",features)
#__________________________scaling_______________________
#__________________training and testing the Algorithme 1______________________
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.30)
clf=RandomForestClassifier(n_estimators=1)
clf=clf.fit(x_train,y_train)
#__________________training and testing the Algorithme 2______________________
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.30)
clf=RandomForestClassifier(n_estimators=2)
clf=clf.fit(x_train,y_train)
 #__________________training and testing the Algorithme 3______________________
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.30)
clf=RandomForestClassifier(n_estimators=3)
clf=clf.fit(x_train,y_train)
 #__________________training and testing the Algorithme 4______________________
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.30)
clf=RandomForestClassifier(n_estimators=4)
clf=clf.fit(x_train,y_train)
#__________________training and testing the Algorithme 5______________________
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.30)
clf=RandomForestClassifier(n_estimators=5)
clf=clf.fit(x_train,y_train)
#__________________ score of training____________
accuracyTrain=clf.score(x_train,y_train)
print("\nthe accuracy Train is\n",accuracyTrain)
#____________score of testing______________
accuracyTest=clf.score(x_test,y_test)
print("\nthe accuracy of test is \n",accuracyTest)
#___________performance appraisal of traning_________________
ypredict=clf.predict(x_train)
print("\n Training Classification report \n",classification_report(y_train,ypredict)
      ,"\nConfusion matrix of training \n",confusion_matrix(y_train,ypredict))
#___________performance appraisal of testing_________________
ypredict=clf.predict(x_test)
print("\n Testing Classification report \n",classification_report(y_test,ypredict)
      ,"\nConfusion matrix of testing \n",confusion_matrix(y_test,ypredict))
#________________database info________________
#df.info()
#sb.countplot(x="Outcome",data=df,palette="hls")
#plt.show()
#sb.countplot(x="Pregnancies",data=df,palette="hls")
#plt.show()
#sb.countplot(x="Glucose",data=df,palette="hls")
#plt.show()
#sb.heatmap(df.corr())
#plt.show()














