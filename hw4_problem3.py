#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample code of HW4, Problem 3
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy import linalg
import math

myfile = open('hw4_p3_data.pickle', 'rb')
mydict = pickle.load(myfile)

X_train = mydict['X_train']
X_test = mydict['X_test']
Y_train = mydict['Y_train']
Y_test = mydict['Y_test'] 

predictive_mean = np.empty(X_test.shape[0])
predictive_std = np.empty(X_test.shape[0])

sigma = 0.1
sigma_f = 1.0
ls = 0.06

#-------- Your code (~10 lines) ---------

covariance_train = [[0] * X_train.shape[0] for i in range(X_train.shape[0])]
covariance_test = [[0] * X_train.shape[0] for i in range(X_test.shape[0])]
for i in range(X_train.shape[0]):
	for j in range(X_train.shape[0]):
		if i == j:
			covariance_train[i][j] = sigma_f ** 2 + sigma ** 2
		else:
			covariance_train[i][j] = sigma_f ** 2 * math.exp(-1 * (((X_train[i] - X_train[j])**2 / (2 * ls ** 2))))

for i in range(X_test.shape[0]):
	for j in range(X_train.shape[0]):
		covariance_test[i][j] = sigma_f ** 2 * math.exp(-1 * (((X_test[i] - X_train[j]) **2 / (2 * ls ** 2))))

for i in range(X_test.shape[0]):
	inv = np.linalg.inv(covariance_train[:][:] + sigma ** 2 * np.identity(X_train.shape[0]))
	y1toN  = np.matmul(inv,np.transpose(Y_train[:]))
	predictive_mean[i] = np.matmul(covariance_test[i][:],y1toN)
	predictive_std[i] = sigma_f ** 2 + sigma ** 2 - np.matmul(covariance_test[i][:],np.matmul(inv,np.transpose(covariance_test[i][:])))

for i in range(Y_test.shape[0]):
	print("Y_t", Y_test[i])
	print("PT", predictive_mean[i])





	 

    
#---------- End of your code -----------

# Optional: Visualize the training data, testing data, and predictive distributions
fig = plt.figure()
plt.plot(X_train, Y_train, linestyle='', color='b', markersize=3, marker='+',label="Training data")
plt.plot(X_test, Y_test, linestyle='', color='orange', markersize=2, marker='^',label="Testing data")
plt.plot(X_test, predictive_mean, linestyle=':', color='green')
plt.fill_between(X_test.flatten(), predictive_mean - predictive_std, predictive_mean + predictive_std, color='green', alpha=0.13)
plt.fill_between(X_test.flatten(), predictive_mean - 2*predictive_std, predictive_mean + 2*predictive_std, color='green', alpha=0.07)
plt.fill_between(X_test.flatten(), predictive_mean - 3*predictive_std, predictive_mean + 3*predictive_std, color='green', alpha=0.04)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
