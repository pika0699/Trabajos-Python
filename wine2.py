# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 08:51:07 2021

@author: andre
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
raw_data = datasets.load_wine()
raw_data
for key,value in raw_data.items():
    print(key,'\n',value,'\n')
print('data.shape\t',raw_data['data'].shape,
      '\ntarget.shape \t',raw_data['target'].shape)
features = pd.DataFrame(data=raw_data['data'],columns=raw_data['feature_names'])
data = features
data['target']=raw_data['target']
data['class']=data['target'].map(lambda ind: raw_data['target_names'][ind])
data.head()    
data.describe()
sns.distplot(data['alcohol'],kde=0)
for i in data.target.unique():
    sns.distplot(data['alcohol'][data.target==i],
                 kde=1,label='{}'.format(i))

plt.legend()
import matplotlib.gridspec as gridspec
for feature in raw_data['feature_names']:
    print(feature)
    #sns.boxplot(data=data,x=data.target,y=data[feature])
    gs1 = gridspec.GridSpec(3,1)
    ax1 = plt.subplot(gs1[:-1])
    ax2 = plt.subplot(gs1[-1])
    gs1.update(right=0.60)
    sns.boxplot(x=feature,y='class',data=data,ax=ax2)
    sns.kdeplot(data[feature][data.target==0],ax=ax1,label='0')
    sns.kdeplot(data[feature][data.target==1],ax=ax1,label='1')
    sns.kdeplot(data[feature][data.target==2],ax=ax1,label='2')
    ax2.yaxis.label.set_visible(False)
    ax1.xaxis.set_visible(False)
    plt.show()
    

from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = \
    train_test_split(raw_data['data'],raw_data['target'],
                     test_size=0.2)
print(len(data_train),' samples in training data\n',
      len(data_test),' samples in test data\n', )
from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn import tree
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.ensemble import RandomForestClassifier
dict_classifiers = { "Nearest Neighbors": {'classifier': KNeighborsClassifier(),'params':[{'n_neighbors': [1, 3, 5, 10],'leaf_size': [3, 30]}]},"Naive Bayes": {'classifier': GaussianNB(),'params': {} }}
from sklearn.model_selection import learning_curve 
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.6, 1.0, 5)):    
 plt.figure()
 plt.title(title)
 if ylim is not None:
   plt.ylim(*ylim)
 plt.xlabel("Training examples")
 plt.ylabel("Score")
 train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
 train_scores_mean = np.mean(train_scores, axis=1)
 train_scores_std = np.std(train_scores, axis=1)
 test_scores_mean = np.mean(test_scores, axis=1)
 test_scores_std = np.std(test_scores, axis=1)
 plt.grid()

 plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
 plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
 plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
 plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

 plt.legend(loc="best")
 return plt    
import time
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score
num_classifiers = len(dict_classifiers.keys())

def batch_classify(X_train, Y_train, X_test, Y_test, verbose = True):
    df_results = pd.DataFrame(data=np.zeros(shape=(num_classifiers,4)),columns = ['classifier','train_score','test_score','training_time'])
    count = 0
    for key, classifier in dict_classifiers.items():
        # t_start = time.clock()
        grid = GridSearchCV(classifier['classifier'],classifier['params'], refit=True,cv = 10,scoring = 'accuracy',n_jobs = -1)
        estimator = grid.fit(X_train,Y_train)
        # t_end = time.clock()
        # t_diff = t_end - t_start
        train_score = estimator.score(X_train,Y_train)
        test_score = estimator.score(X_test,Y_test)
        df_results.loc[count,'classifier'] = key
        df_results.loc[count,'train_score'] = train_score
        df_results.loc[count,'test_score'] = test_score
        # df_results.loc[count,'training_time'] = t_diff
        # if verbose:
        #     print("trained {c} in {f:.2f} s".format(c=key,
        #                                             f=t_diff))
        count+=1
        plot_learning_curve(estimator, 
                              "{}".format(key),
                              X_train,
                              Y_train,
                              ylim=(0.75,1.0),
                              cv=10)
    return df_results

df_results = batch_classify(data_train, label_train, data_test, label_test)
display(df_results.sort_values(by='test_score', ascending=False))
