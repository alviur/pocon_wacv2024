#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:55:00 2023

@author: agomezvi
"""

import numpy as np
from matplotlib import pyplot as plt

## # Platicity - 4 tasks all methods
#pocon_expert_4tasks_scratchOP = [86.8, 85.6, 80.9, 76.8]
#pocon_expert_4tasks_ftOP = [85.1, 82.0, 78.2, 77.2]
#pocon_expert_4tasks_copyOP = [86.5, 85.8, 82.39999999999999, 79.10000000000001]
#pocon_expert_4tasks_d2eOP = []
#
#
## Platicity - 4 tasks all methods
#pocon_main_4tasks_scratchOP = [87.0, 89.8, 85.39999999999999, 79.7]
#pocon_main_4tasks_ftOP = [87.0, 89.2, 86.3, 83.6]
#pocon_main_4tasks_copyOP = [87.0, 89.4, 86.5, 84.3]
#pocon_main_4tasks_d2eOP = []
#
#plt.plot(pocon_expert_4tasks_scratchOP, 'b-', label='scratchOP - exp')
#plt.plot(pocon_main_4tasks_scratchOP, 'b--', label='scratchOP - main')
#plt.plot(pocon_expert_4tasks_ftOP, 'c-', label='ftOP - exp')
#plt.plot(pocon_main_4tasks_ftOP, 'c--', label='ftOP - main')
#plt.plot(pocon_expert_4tasks_copyOP, 'g-', label='copyOP - exp')
#plt.plot(pocon_main_4tasks_copyOP, 'g--', label='copyOP - main')
##plt.plot(pocon_expert_4tasks_d2eOP, 'r-', label='d2eOP - exp')
##plt.plot(pocon_main_4tasks_d2eOP, 'r--', label='d2eOP - main')
#plt.title('Last task performance (plasticity) - 4 tasks')
#plt.grid()
#plt.legend()



# # Platicity - 10 tasks all methods
#pocon_expert_10tasks_scratchOP = [82.8, 81.69999999999999, 70.5, 70.3, 65.4, 80.60000000000001, 69.19999999999999, 64.8, 72.6, 71.1]
#pocon_expert_10tasks_ftOP = [82.8, 86.5, 81.10000000000001, 79.5, 77.0, 82.19999999999999, 81.0, 76.1, 83.8, 83.2]
#pocon_expert_10tasks_copyOP = [82.8, 88.2, 83.0, 80.4, 79.3, 86.4, 83.6, 82.0, 88.2, 85.6]
#pocon_expert_10tasks_d2eOP = [82.8, 88.2, 80.6, 77.7, 74.9, 84.0, 80.5, 77.8, 86.3, 82.3]
#
#
#
## Platicity - 4 tasks all methods
#pocon_main_10tasks_scratchOP = [83.0, 84.5, 75.7, 73.6, 70.1, 81.2, 75.4, 69.6, 77.5, 75.2]
#pocon_main_10tasks_ftOP = [83.0, 87.6, 83.5, 81.10000000000001, 78.8, 87.2, 84.2, 80.4, 86.9, 86.0]
#pocon_main_10tasks_copyOP = [83.0, 88.6, 84.2, 81.5, 78.4, 86.1, 84.0, 80.0, 86.5, 84.3]
#pocon_main_10tasks_d2eOP = [83.0, 87.7, 84.5, 80.60000000000001, 79.3, 86.2, 84.1, 81.69999999999999, 85.8, 86.4]
#
#
#plt.plot(pocon_expert_10tasks_scratchOP, 'b-', label='scratchOP - exp')
#plt.plot(pocon_main_10tasks_scratchOP, 'b--', label='scratchOP - main')
#plt.plot(pocon_expert_10tasks_ftOP, 'c-', label='ftOP - exp')
#plt.plot(pocon_main_10tasks_ftOP, 'c--', label='ftOP - main')
#plt.plot(pocon_expert_10tasks_copyOP, 'g-', label='copyOP - exp')
#plt.plot(pocon_main_10tasks_copyOP, 'g--', label='copyOP - main')
#plt.plot(pocon_expert_10tasks_d2eOP, 'r-', label='d2eOP - exp')
#plt.plot(pocon_main_10tasks_d2eOP, 'r--', label='d2eOP - main')
#plt.title('Last task performance (plasticity) - 10 tasks')
#plt.grid()
#plt.legend()


# # Platicity - 10 tasks all methods - FTop
#pocon_expert_10tasks_ftOP_r18_r18 = [82.8, 86.5, 81.10000000000001, 79.5, 77.0, 82.19, 81.0, 76.1, 83.8, 83.2]
#pocon_expert_10tasks_ftOP_r18_r9 = [84.39, 88.2, 82.89, 80.60, 76.9, 85.9, 82.19, 79.2, 86.0, 85.1]
#pocon_expert_10tasks_ftOP_r9_r9 = [84.39, 87.6, 84.899, 81.2, 78.7, 85.0, 83.0, 78.3, 86.8, 84.1]
#
#
## Platicity - 10 tasks all methods
#pocon_main_10tasks_ftOP_r18_r18 = [83.0, 87.6, 83.5, 81.10000000000001, 78.8, 87.2, 84.2, 80.4, 86.9, 86.0]
#pocon_main_10tasks_ftOP_r18_r9 = [84.39, 87.1, 82.0, 76.9, 76.6, 85.5, 82.199, 80.2, 84.2, 84.1]
#pocon_main_10tasks_ftOP_r9_r9 = [86.2, 87.8, 81.10, 77.7, 74.8, 85.8, 81.89, 76.4, 83.0, 83.1]
#
#
#plt.plot(pocon_expert_10tasks_ftOP_r18_r18, 'c-', label='R18-R18 - exp')
#plt.plot(pocon_main_10tasks_ftOP_r18_r18, 'c--', label='R18-R18  - main')
#plt.plot(pocon_expert_10tasks_ftOP_r18_r9, 'r-', label='R18-R9  - exp')
#plt.plot(pocon_main_10tasks_ftOP_r18_r9, 'r--', label='R18-R9  - main')
#plt.plot(pocon_expert_10tasks_ftOP_r9_r9, 'g-', label='R9-R9  - exp')
#plt.plot(pocon_main_10tasks_ftOP_r9_r9, 'g--', label='R9-R9  - main')
#
#
#plt.title('Last task performance (plasticity) - FtOP 10 tasks')
#plt.grid()
#plt.legend(loc='lower left')


 # Platicity - 4 tasks all methods - FTop
pocon_expert_10tasks_ftOP_r18_r18 = [85.1, 82.0, 78.2, 77.2]
pocon_expert_10tasks_ftOP_r18_r9 = [87.2, 86.6, 83.39, 78.5]
pocon_expert_10tasks_ftOP_r9_r9 = [87.9, 87.7, 82.0, 78.4]


# Platicity - 4 tasks all methods
pocon_main_10tasks_ftOP_r18_r18 = [85.1, 89.2, 86.3, 83.6]
pocon_main_10tasks_ftOP_r18_r9 = [87.2, 89.7, 84.39, 79.2]
pocon_main_10tasks_ftOP_r9_r9 = [87.9, 89.8, 84.7, 78.2]


plt.plot(pocon_expert_10tasks_ftOP_r18_r18, 'c-', label='R18-R18 - exp')
plt.plot(pocon_main_10tasks_ftOP_r18_r18, 'c--', label='R18-R18  - main')
plt.plot(pocon_expert_10tasks_ftOP_r18_r9, 'r-', label='R18-R9  - exp')
plt.plot(pocon_main_10tasks_ftOP_r18_r9, 'r--', label='R18-R9  - main')
plt.plot(pocon_expert_10tasks_ftOP_r9_r9, 'g-', label='R9-R9  - exp')
plt.plot(pocon_main_10tasks_ftOP_r9_r9, 'g--', label='R9-R9  - main')


plt.title('Last task performance (plasticity) - FtOP 4 tasks')
plt.grid()
plt.legend(loc='lower left')

## # Platicity - 20 tasks all methods
#pocon_expert_20tasks_scratchOP = []
#pocon_expert_20tasks_ftOP = []
#pocon_expert_20tasks_copyOP = []
#pocon_expert_20tasks_d2eOP = []
#
#
## Platicity - 4 tasks all methods
#pocon_main_20tasks_scratchOP = []
#pocon_main_20tasks_ftOP = []
#pocon_main_20tasks_copyOP = []
#pocon_main_20tasks_d2eOP = []
#
#plt.plot(pocon_expert_20tasks_scratchOP, 'b-', label='scratchOP - exp')
#plt.plot(pocon_main_20tasks_scratchOP, 'b--', label='scratchOP - main')
#plt.plot(pocon_expert_20tasks_ftOP, 'c-', label='ftOP - exp')
#plt.plot(pocon_main_20tasks_ftOP, 'c--', label='ftOP - main')
#plt.plot(pocon_expert_20tasks_copyOP, 'g-', label='copyOP - exp')
#plt.plot(pocon_main_20tasks_copyOP, 'g--', label='copyOP - exp')
##plt.plot(pocon_expert_20tasks_d2eOP, 'r-', label='d2eOP - exp')
##plt.plot(pocon_main_20tasks_d2eOP, 'r--', label='d2eOP - main')
#plt.title('Last task performance (plasticity) - 20 tasks')
#plt.grid()
#plt.legend()