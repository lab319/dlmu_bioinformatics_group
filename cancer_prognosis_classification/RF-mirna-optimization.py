# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:00:44 2019

@author: lab319
"""

import numpy as np 
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler   
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

file_1=open("KIRP_mirna.txt")
file_1.readline()
x=np.loadtxt(file_1)
file_1.close()

file_2=open("KIRP_label.txt")
file_2.readline()
y=np.loadtxt(file_2)
file_2.close()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y,random_state=1)
   
ss=StandardScaler()
x_train_ss=ss.fit_transform(x_train)
x_test_ss=ss.transform(x_test)
param_grid = {"n_estimators": [100,500,1000,1500,2000,2500]}
clf = RandomForestClassifier(random_state=0)
grid = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc',verbose=True)
grid.fit(x_train_ss, y_train)
score_train=grid.best_score_
print ("grid.best_score_",score_train)
print ("grid.best_params_",grid.best_params_)

#('grid2.best_score_', 0.8215331684971156)
#('grid2.best_params_', {'n_estimators': 100})


