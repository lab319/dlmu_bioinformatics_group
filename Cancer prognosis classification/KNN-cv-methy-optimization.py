# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:35:40 2019

@author: Administrator
"""

import numpy as np  
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file_1=open("KIRP_methy.txt")
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

param_digits =[
    {
        'weights':['uniform'],
        'n_neighbors':[i for i in range(1,11)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,6)],
        'p':[i for i in range(1,6)]
    }
]

clf = KNeighborsClassifier()
 
grid = GridSearchCV(clf,param_digits,cv=5,scoring='roc_auc',verbose=True)
grid.fit(x_train_ss, y_train)
score_train=grid.best_score_
print ("grid2.best_score_",score_train)
print ("grid2.best_params_",grid.best_params_)

#('grid2.best_score_', 0.678215743775516)
#('grid2.best_params_', {'n_neighbors': 8, 'weights': 'uniform'})


