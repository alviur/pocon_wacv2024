# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:29:37 2023

@author: Alex
plot Dirichlet  partition of num_task from num_clases
"""

import numpy as np
import random
import matplotlib.pyplot as plt

num_task = 5
num_clases = 10
num_samples = 1000
alpha = 0.5

sample_matrix = num_samples*np.random.dirichlet(np.repeat(0.5, num_task),num_clases)
acumm = np.zeros_like(sample_matrix[:,0])
for i in range(num_task):
         
    plt.barh(range(num_clases), sample_matrix[:,i], left=acumm)    
    acumm += sample_matrix[:,i]
    
plt.title("Dirichlet partition")
plt.xlabel('Partition - each color is a task')
plt.ylabel('Class label')

print(sample_matrix.astype(int))
# num_samples = 10
# from numpy.random import multinomial
# np.random.multinomial(num_samples, np.ones(num_task)/num_task)