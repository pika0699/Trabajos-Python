# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:14:54 2021

@author: Pika
"""

import seaborn as sns
import pandas as pd

from sklearn import datasets
from IPython.display import display

from sklearn import (datasets, metrics,
                     model_selection as skms,
                     naive_bayes, neighbors)

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data,
                       columns = iris.feature_names)
iris_df['target'] = iris.target
display(pd.concat([iris_df.head(3),
                   iris_df.tail(3)]))
sns.pairplot(iris_df, hue = 'target', height=1.5)
print('targets: {}'.format(iris.target_names), iris.target_names[1], sep='\n')















