# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:00:44 2019

@author: lab319
"""

import numpy as np
import xgboost as xgb
from sklearn import metrics 
from xgboost.sklearn import XGBClassifier  
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split

file_1=open("KIRP_mrna.txt")
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
print('train.shape',x_train.shape,'test.shape', x_test.shape)

xgb1= XGBClassifier(
        learning_rate =0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=1)

xgb_param = xgb1.get_xgb_params()
xgtrain = xgb.DMatrix(x_train_ss, label=y_train)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
    metrics='auc', early_stopping_rounds=50,verbose_eval=10)

print('n_estimators',cvresult.shape[0])
print('test-auc:',cvresult.iloc[cvresult.shape[0]-1, 0])
xgb1.set_params(n_estimators=cvresult.shape[0])
print('model',xgb1)

#n_estimators 123
#test-auc: 0.8937476

tuned_parameters = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
xgb2 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=123,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=1,
        nthread=6)

grid_2 = GridSearchCV(xgb2, param_grid=tuned_parameters, cv=5,scoring='roc_auc',verbose=True)
grid_2.fit(x_train_ss, y_train)
score_train=grid_2.best_score_
print ("clf.best_score_",score_train)
print ("clf.best_params_",grid_2.best_params_)

#clf.best_score_ 0.8687213692906293
#clf.best_params_ {'max_depth': 4, 'min_child_weight': 3}


tuned_parameters = {'gamma': [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
xgb3 =  XGBClassifier(
        learning_rate =0.1,
        n_estimators=123,
        max_depth=4,
        min_child_weight=3,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=1,
        nthread=6)

grid_3 = GridSearchCV(xgb3, param_grid=tuned_parameters, cv=5,scoring='roc_auc',verbose=True)
grid_3.fit(x_train_ss, y_train)
score_train=grid_3.best_score_
print ("clf.best_score_",score_train)
print ("clf.best_params_",grid_3.best_params_)

#clf.best_score_ 0.8687213692906293
#clf.best_params_ {'gamma': 0}


tuned_parameters = {'subsample': [0.6, 0.7, 0.8, 0.9,1.0], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9,1.0]}
xgb4 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=123,
        max_depth=4,
        min_child_weight=3,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=1,
        nthread=6)
grid_4 = GridSearchCV(xgb4, param_grid=tuned_parameters, cv=5,scoring='roc_auc',verbose=True)
grid_4.fit(x_train_ss, y_train)
score_train=grid_4.best_score_
print ("clf.best_score_",score_train)
print ("clf.best_params_",grid_4.best_params_)

#clf.best_score_ 0.8736271634753606
#clf.best_params_ {'colsample_bytree': 0.9, 'subsample': 0.6}

tuned_parameters = {'reg_alpha': [0,0.05, 0.1, 1, 2, 3], 'reg_lambda': [0,0.05, 0.1, 1, 2, 3]}
xgb5 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=123,
        max_depth=4,
        min_child_weight=3,
        gamma=0,
        subsample=0.6,
        colsample_bytree=0.9,
        seed=1,
        nthread=6)

grid_5 = GridSearchCV(xgb5, param_grid=tuned_parameters, cv=5,scoring='roc_auc',verbose=True)
grid_5.fit(x_train_ss, y_train)
score_train=grid_5.best_score_
print ("clf.best_score_",score_train)
print ("clf.best_params_",grid_5.best_params_)

#clf.best_score_ 0.88191880857915
#clf.best_params_ {'reg_alpha': 0.1, 'reg_lambda': 0}

xgb6 = XGBClassifier(
        learning_rate =0.01,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=3,
        gamma=0,
        subsample=0.6,
        colsample_bytree=0.9,
        seed=1,
        reg_alpha= 0.1, 
        reg_lambda= 0,
        nthread=6)

xgb_param = xgb6.get_xgb_params()
xgtrain = xgb.DMatrix(x_train_ss, label=y_train)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb6.get_params()['n_estimators'], nfold=5,
    metrics='auc', early_stopping_rounds=50,verbose_eval=10)

print('test-auc:',cvresult.iloc[cvresult.shape[0]-1, 0])
xgb6.set_params(n_estimators=cvresult.shape[0])
print('model',xgb6)

