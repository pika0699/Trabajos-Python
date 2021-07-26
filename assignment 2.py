# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:42:56 2021

@author: Pika
"""


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score



from sklearn import datasets
from IPython.display import display

from sklearn import (datasets, metrics,
                     model_selection as skms,
                     naive_bayes, neighbors)
import numpy as np #Algebra lineal
import pandas as pd # Procesamiento de Datos
import matplotlib.pyplot as plt # Visualización
import seaborn as sns # Visualización4
import scipy.stats as stats


from sklearn.datasets import load_breast_cancer



cancer = datasets.load_breast_cancer()
cancer_df = pd.DataFrame(data=cancer['data'],columns=cancer['feature_names'])



data = cancer_df
data['target'] = cancer['target']
data['class']=data['target'].map(lambda ind: cancer['target_names'][ind])
#sns.pairplot(cancer_df, hue = 'target', height=1.5)
print('targets: {}'.format(cancer.target_names), cancer.target_names[1], sep='\n')

# name of features
print(cancer['feature_names'])
# description of data
print(cancer['DESCR'])

# =============================================================================
# 
# # Count the target class
# sns.countplot(cancer_df['target'])
# 
# # pair plot of sample feature
# sns.pairplot(cancer_df, hue = 'target', 
#              vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area'] )
# 
# 
# sns.pairplot(cancer_df, hue = 'target', 
#              vars = ['mean smoothness', 'mean compactness', 'mean concavity', 'mean symmetry'] )
# 
# 
# =============================================================================
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 

# =============================================================================
# 
# sns.displot(cancer_df['mean area'][cancer.target==1],
#             kde=1, label='1', color = 'r')
# sns.displot(cancer_df['mean area'][cancer.target==0],
#             kde=1, label='{}'.format(0))
# plt.legend()
# 
# 
# =============================================================================


# =============================================================================
# plt.figure(1)
# 
# sns.kdeplot(cancer_df['mean radius'][cancer.target==0],label = '0')
# 
# sns.kdeplot(cancer_df['mean radius'][cancer.target==1],label = '1')
# 
# plt.figure(2)
# 
# sns.boxplot(x=cancer_df['mean radius'],y='class',data=data)
# 
# =============================================================================

# =============================================================================
# 
# import matplotlib.gridspec as gridspec
# 
# for cancer_df in cancer['feature_names']:
#     plt.figure(cancer_df)
#     print(cancer_df)
#     #sns.boxplot(data=data,x=cancer.target,y=data[cancer_df])
#     gs1 = gridspec.GridSpec(2,1)
#     ax1 = plt.subplot(gs1[:-1])
#     ax2 = plt.subplot(gs1[-1])
#     gs1.update(right=0.80)
#     sns.boxplot(x=cancer_df,y='class',data=data,ax=ax2)
#     sns.kdeplot(data[cancer_df][cancer.target==0],ax=ax1,label='0')
#     sns.kdeplot(data[cancer_df][cancer.target==1],ax=ax1,label='1')
#     ax2.yaxis.label.set_visible(False)
#     ax1.xaxis.set_visible(False)
#     plt.show()
# 
# 
# sns.pairplot(cancer_df, hue='class')
# 
# =============================================================================

# =============================================================================
# X = cancer_df.drop(['target'], axis = 1)
# y = cancer_df['target']
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)
# 
# =============================================================================


# =============================================================================
# # Heatmap of Correlation matrix of breast cancer DataFrame
# plt.figure(figsize=(15,15))
# sns.heatmap(cancer_df.corr(), annot = True, cmap ='coolwarm', linewidths=2)
# 
# =============================================================================



from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = \
    train_test_split(cancer['data'],cancer['target'],
                     test_size=0.3, random_state=20)
print('Muestras de entrenamiento: ',len(data_train),' \n',
      'Muestras de prueba:',len(data_test),'\n', )



from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()
 
naive_bayes.fit(data_train , label_train)
 
#Predicción de los datos

y_predicted = naive_bayes.predict(data_test)

#Import metrics class from sklearn
from sklearn import metrics
 
print('Precisión porcentual del Naive Bayes = ',metrics.accuracy_score(y_predicted , label_test)*100)

# K – Nearest Neighbor Classifier

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(data_train, label_train)
y_pred_knn = knn_classifier.predict(data_test)

print('Precisión porcentual del Knn = ',metrics.accuracy_score(y_pred_knn,label_test)*100)


# =============================================================================
# 
# from sklearn.model_selection import learning_curve 
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.6, 1.0, 5)):
# 
# 
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
# 
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
# 
#     plt.legend(loc="best")
#     return plt
# 
# =============================================================================
















# Normalización de datos a Zscore


data_train_Z = (data_train-data_train.mean())/(data_train.std())
data_test_Z = (data_test-data_train.mean())/(data_test.std())

label_train_Z = (label_train-label_train.mean())/(label_train.std())
label_test_Z = (label_test-label_train.mean())/(label_test.std())



naive_bayes1 = GaussianNB()
 
naive_bayes1.fit(data_train_Z , label_train_Z)
 
#Predicción de los datos
y_predicted1 = naive_bayes1.predict(data_test_Z)


 
#print('Precisión del Naive Bayes = ',metrics.accuracy_score(y_predicted1 , label_test_Z))







# =============================================================================
# sns.displot(X_train_N['mean area'][cancer.target==0],kde =0
#             , color = 'r')
# 
# sns.displot()
# =============================================================================
# =============================================================================
# # Feature scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train_sc = sc.fit_transform(X_train)
# X_test_sc = sc.transform(X_test)
# 
# =============================================================================

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score




# =============================================================================
# # K – Nearest Neighbor Classifier
# from sklearn.neighbors import KNeighborsClassifier
# knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# knn_classifier.fit(X_train, y_train)
# y_pred_knn = knn_classifier.predict(X_test)
# accuracy_score(y_test, y_pred_knn)
# 
# =============================================================================




# =============================================================================
# 
# def setUp(self):
#         self.roc_floor = 0.9
#         self.accuracy_floor = 0.9
# 
#         random_state = 42
#         X, y = load_breast_cancer(return_X_y=True)
# 
#         self.X_train, self.X_test, self.y_train, self.y_test = \
#             train_test_split(X, y, test_size=0.4, random_state=random_state)
# 
#         classifiers = [DecisionTreeClassifier(random_state=random_state),
#                        LogisticRegression(random_state=random_state),
#                        KNeighborsClassifier(),
#                        RandomForestClassifier(random_state=random_state),
#                        GradientBoostingClassifier(random_state=random_state)]
# 
#         self.clf = DES_LA(classifiers, local_region_size=30)
#         self.clf.fit(self.X_train, self.y_train) 
# 
# 
# 
# =============================================================================







# =============================================================================
# 
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC, LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn import tree
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.ensemble import RandomForestClassifier
# 
# dict_classifiers = {
#     "Logistic Regression": 
#             {'classifier': LogisticRegression(),
#                 'params' : [
#                             {
#                              'penalty': ['l1','l2'],
#                              'C': [0.001,0.01,0.1,1,10,100,1000]
#                             }
#                            ]
#             },
#     "Nearest Neighbors": 
#             {'classifier': KNeighborsClassifier(),
#                  'params': [
#                             {
#                             'n_neighbors': [1, 3, 5, 10],
#                             'leaf_size': [2, 30]
#                             }
#                            ]
#             },
#              
#     "Linear SVM": 
#             {'classifier': SVC(),
#                  'params': [
#                             {
#                              'C': [1, 10, 100, 1000],
#                              'gamma': [0.001, 0.0001],
#                              'kernel': ['linear']
#                             }
#                            ]
#             },
#     "Gradient Boosting Classifier": 
#             {'classifier': GradientBoostingClassifier(),
#                  'params': [
#                             {
#                              'learning_rate': [0.05, 0.1],
#                              'n_estimators' :[50, 100, 200],
#                              'max_depth':[2,None]
#                             }
#                            ]
#             },
#     "Decision Tree":
#             {'classifier': tree.DecisionTreeClassifier(),
#                  'params': [
#                             {
#                              'max_depth':[2,None]
#                             }
#                              ]
#             },
#     "Random Forest": 
#             {'classifier': RandomForestClassifier(),
#                  'params': {}
#             },
#     "Naive Bayes": 
#             {'classifier': GaussianNB(),
#                  'params': {}
#             }
# }
# 
# 
# 
# # =============================================================================
# # 4
# # =============================================================================
# 
# 
# 
# from sklearn.model_selection import learning_curve 
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.6, 1.0, 5)):
#     """
#     Generate a simple plot of the test and traning learning curve.
# 
#     Parameters
#     ----------
#     estimator : object type that implements the "fit" and "predict" methods
#         An object of that type which is cloned for each validation.
# 
#     title : string
#         Title for the chart.
# 
#     X : array-like, shape (n_samples, n_features)
#         Training vector, where n_samples is the number of samples and
#         n_features is the number of features.
# 
#     y : array-like, shape (n_samples) or (n_samples, n_features), optional
#         Target relative to X for classification or regression;
#         None for unsupervised learning.
# 
#     ylim : tuple, shape (ymin, ymax), optional
#         Defines minimum and maximum yvalues plotted.
# 
#     cv : integer, cross-validation generator, optional
#         If an integer is passed, it is the number of folds (defaults to 3).
#         Specific cross-validation objects can be passed, see
#         sklearn.cross_validation module for the list of possible objects
# 
#     n_jobs : integer, optional
#         Number of jobs to run in parallel (default 1).
#     """
#     plt.figure(3)
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
# 
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
# 
#     plt.legend(loc="best")
#     return plt
# 
# =============================================================================



# =============================================================================
# import time
# from sklearn.model_selection import GridSearchCV 
# from sklearn.metrics import accuracy_score
# num_classifiers = len(dict_classifiers.keys())
# 
# def batch_classify(X_train, Y_train, X_test, Y_test, verbose = True):
#     df_results = pd.DataFrame(
#         data=np.zeros(shape=(num_classifiers,4)),
#         columns = ['classifier',
#                    'train_score', 
#                    'test_score',
#                    'training_time'])
#     count = 0
#     for key, classifier in dict_classifiers.items():
#         t_start = time.clock()
#         grid = GridSearchCV(classifier['classifier'], 
#                       classifier['params'],
#                       refit=True,
#                         cv = 10, # 9+1
#                         scoring = 'accuracy', # scoring metric
#                         n_jobs = -1
#                         )
#         estimator = grid.fit(X_train,
#                              Y_train)
#         t_end = time.clock()
#         t_diff = t_end - t_start
#         train_score = estimator.score(X_train,
#                                       Y_train)
#         test_score = estimator.score(X_test,
#                                      Y_test)
#         df_results.loc[count,'classifier'] = key
#         df_results.loc[count,'train_score'] = train_score
#         df_results.loc[count,'test_score'] = test_score
#         df_results.loc[count,'training_time'] = t_diff
#         if verbose:
#             print("trained {c} in {f:.2f} s".format(c=key,
#                                                     f=t_diff))
#         count+=1
#         plot_learning_curve(estimator, 
#                               "{}".format(key),
#                               X_train,
#                               Y_train,
#                               ylim=(0.75,1.0),
#                               cv=10)
#     return print(df_results)
# 
# =============================================================================
















