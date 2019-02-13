# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:44:00 2019

@author: 687338
"""

import pandas as pd
import numpy as np


df=pd.read_csv("C:\\Users\\687338\\Desktop\\Datasets\\trnscs.csv")
#print(df.head(5))

print(df.columns)

sub_df=df[['Date Failure','H2','CH4','CO','CO2','C2H6','C2H4','C2H2','Total Gas','Failure']]
print(sub_df.head(5))
print(sub_df.describe())

print(sub_df.isnull().any())
sub_df['Total Gas'] = sub_df['Total Gas'].fillna(method='ffill')

corr_matrix = df.corr().abs()
print(corr_matrix)

#===============================================================================================================
from sklearn.model_selection import train_test_split
X=np.asarray(sub_df[['H2','CH4','CO','CO2','C2H6','C2H4','C2H2']])
Y=np.asarray(sub_df[['Failure']])

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
#y_train=y_train.ravel()

#print(y_train.shape)
#================================================================================================================

from sklearn import svm
sv=svm.SVC(kernel='linear')
sv.fit(x_train,y_train)
final_y=sv.predict(x_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,final_y)*100
print("\n",x_test,"\n",y_test,"\n",final_y,"\nThe Svm Accuracy is=",acc)

#================================================================================================================

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(C=0.01, solver='liblinear')
LR.fit(x_train,y_train)
LR_final=LR.predict(x_test)

yhat_prob = LR.predict_proba(x_test)
print(yhat_prob)

print("\nThe LR accuracy",accuracy_score(y_test,LR_final)*100)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=20)
rf.fit(x_train,y_train)
rf_final_y=rf.predict(x_test)

print("The RandomForestClassifier Accuracy is",accuracy_score(y_test,rf_final_y)*100)







