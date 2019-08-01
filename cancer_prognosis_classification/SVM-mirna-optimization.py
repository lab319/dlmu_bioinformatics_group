# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:00:44 2019

@author: lab319
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:23:51 2019

@author: Administrator
"""


import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

param_grid = [{'kernel': ['linear'],'C': [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,0.1, 1.0, 10,1e2,1e3,1e4,1e5,1e6,1e7,1e8]}]
clf = SVC(probability=True)
grid2 = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc',verbose=True)
grid2.fit(x_train_ss, y_train)
score_train=grid2.best_score_
print ("grid2.best_score_",score_train)
print ("grid2.best_params_",grid2.best_params_)

#('grid2.best_score_', 0.8370344814367584)
#('grid2.best_params_', {'kernel': 'linear', 'C': 1e-05})

