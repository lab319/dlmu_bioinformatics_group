# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:35:40 2019

@author: Administrator
"""

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

file_1=open("KIRP_mrna.txt")
file_1.readline()
x=np.loadtxt(file_1)
file_1.close()

file_4=open("KIRP_label.txt")
file_4.readline()
y=np.loadtxt(file_4)
file_4.close()

scores_1=[]
scores_2=[]
scores_3=[]
scores_4=[]
scores_5=[]
scores_6=[]
for i in range(100):
    print('###########Time############',i+1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y,random_state=i)
   
    ss=StandardScaler()
    x_train_ss=ss.fit_transform(x_train)
    x_test_ss=ss.transform(x_test)

    model= KNeighborsClassifier(weights= 'distance',n_neighbors= 5,p= 1) 
    model.fit(x_train_ss, y_train)  
    y_proba=model.predict_proba(x_test_ss)[:,1]
    y_pred = model.predict(x_test_ss)
    
    auc = metrics.roc_auc_score(y_test, y_proba) 
    acc = accuracy_score(y_test, y_pred)
    
    precision, recall, _thresholds = metrics.precision_recall_curve(y_test, y_proba)
    pr_auc = metrics.auc(recall, precision)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total=tn+fp+fn+tp
    sen = float(tp)/float(tp+fn)
    sps = float(tn)/float((tn+fp))
    
     print ('AUC: %f' % auc)
    print ('ACC: %f' % acc) 
    print("PRAUC: %f" % pr_auc)
    print ('MCC : %f' % mcc)
    print ('SEN: %f' % sen)
    print ('SEP: %f' % sps)
    
    scores_1.append(auc)
    scores_2.append(acc)
    scores_3.append(pr_auc)
    scores_4.append(mcc)
    scores_5.append(sen)
    scores_6.append(sps)
    i=i+1
    
    score_auc=str(auc)+'\n'           
    df_1=open('kirp-knn-aucs-mrna.txt','a')
    df_1.write(score_auc)
    
    score_acc=str(acc)+'\n'           
    df_1=open('kirp-knn-accs-mrna.txt','a')
    df_1.write(score_acc)
    
    score_pr_auc=str(pr_auc)+'\n'           
    df_1=open('kirc-knn-prauc-mrna.txt','a')
    df_1.write(score_pr_auc)
    
    score_mcc=str(mcc)+'\n'           
    df_1=open('kirp-knn-mcc-mrna.txt','a')
    df_1.write(score_mcc)
   
    score_sen=str(sen)+'\n'           
    df_1=open('kirp-knn-sen-mrna.txt','a')
    df_1.write(score_sen)
    
    score_sps=str(sps)+'\n'           
    df_1=open('kirp-knn-sps-mrna.txt','a')
    df_1.write(score_sps)
    
print('auc-mean-score: %.3f' %np.mean(scores_1))
print('acc-mean-score: %.3f' %np.mean(scores_2))
print('pr-mean-score: %.3f' %np.mean(scores_3))
print('mcc-mean-score: %.3f' %np.mean(scores_4))
print('sen-mean-score: %.3f' %np.mean(scores_5))
print('sps-mean-score: %.3f' %np.mean(scores_6))

'''
auc-mean-score: 0.665
acc-mean-score: 0.773
pr-mean-score: 0.871
mcc-mean-score: 0.263
sen-mean-score: 0.984
sps-mean-score: 0.158
'''


