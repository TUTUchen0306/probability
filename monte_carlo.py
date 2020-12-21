import numpy as np
import math


def func(x):
	return math.cos(x) + math.sin(2*x);

def monte_carlo(n_sample, mu, sigma):
	val_lst = []
	for i in range(n_sample):
		x = np.random.normal(mu, sigma)
		val_lst.append(func(x))
	return np.mean(val_lst)

time = 20
MAX_SAMPLE = 100000
for i in range(time):
	x = monte_carlo(MAX_SAMPLE, 0, 1)
	print(x)
