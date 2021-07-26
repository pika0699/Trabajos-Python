# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:37:13 2021

@author: Pika
"""

import numpy as np
from scipy import linalg, sparse
n = 3
N = 100
on = np.ones((1,N))
x = np.random.random((1,N))
y = np.random.random((3,N))
Xaug = np.concatenate((x,on))
Xaug =Xaug.T
y = y.T
betas = np.matmul(np.matmul(linalg.inv(np.matmul(Xaug.T,Xaug)),
                            Xaug.T),y)

print(betas)




