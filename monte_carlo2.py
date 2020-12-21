import numpy as np
import math

def func(x, y):
	if x * x + 2 * y * y <= 1:
		return 1
	else:
		return 0

def monte_carlo(n_sample):
	val_list = []
	for i in range(n_sample):
		x = np.random.uniform(-1,1);
		y = np.random.uniform(-1,1);
		val_list.append(func(x,y))
	return np.mean(val_list) * 4

MAX_SAMPLE = 10000000
x = monte_carlo(MAX_SAMPLE)
print(x)
