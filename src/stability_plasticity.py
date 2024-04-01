# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 06:57:21 2023

@author: Alex
"""

import numpy as np
from matplotlib import pyplot as plt

# #stability
# pocon_main_4tasks = [76.12, 80.24, 81.64, 82.44]
# pocon_expert_4tasks = [74.03999999999999, 71.72, 71.8, 75.56]
# pfr_4tasks = [75.6, 77.28, 77.44, 79.32]

# plt.plot(pocon_main_4tasks, label='POCON (main)')
# plt.plot(pocon_expert_4tasks, label='POCON (expert)')
# plt.plot(pfr_4tasks, label='PFR')
# plt.title('First task performance (stability)')
# plt.grid()
# plt.legend()


# # Platicity - baseline
# pocon_main_4tasks = [76.12, 67.36, 76.2, 77.28]
# pocon_expert_4tasks = [74.72, 61.76, 70.52, 66.47]
# pfr_4tasks = [75.72, 66.679, 76.72, 77.56]

# plt.plot(pocon_main_4tasks, label='POCON (main)')
# plt.plot(pocon_expert_4tasks, label='POCON (expert)')
# plt.plot(pfr_4tasks, label='PFR')
# plt.title('Last task performance (plasticity)')
# plt.grid()
# plt.legend()

# [73.4, 60.6, 70.76]

# # Platicity - 500 epochs train 
# pocon_main_4tasks = []
# pocon_expert_4tasks = [73.4, 60.6, 70.76]
# pfr_4tasks = [75.72, 66.679, 76.72, 77.56]


# plt.plot(pocon_main_4tasks, label='POCON (main)')
# plt.plot(pocon_expert_4tasks, label='POCON (expert)')
# plt.plot(pfr_4tasks, label='PFR')
# plt.title('Last task performance (plasticity)')
# plt.grid()
# plt.legend()

# # Platicity - 500 epochs train 
# pocon_main_4tasks = [80.44, 70.39999999999999, 77.16, 76.52]
# pocon_expert_4tasks = [79.56, 69.04, 75.56, 78.44]
# pfr_4tasks = [75.72, 66.679, 76.72, 77.56]

# pocon_main_4tasks = [79.60000000000001, 69.44, 79.0, 77.8]
# pocon_expert_4tasks = [76.84, 63.2, 70.8, 68.4]

# plt.plot(pocon_main_4tasks, label='POCON (main)')
# plt.plot(pocon_expert_4tasks, label='POCON (expert)')
# plt.plot(pfr_4tasks, label='PFR')
# plt.title('Last task performance (plasticity) FT')
# plt.grid()
# plt.legend()

# #############################################################
# ##################### 4 tasks ###############################

# # Platicity - 4 tasks all methods
pocon_expert_4tasks_scratchOP = [76.08, 63.239999999999995, 70.84, 68.36]
pocon_expert_4tasks_ftOP_1500 = [87.7, 86.7, 82.0, 77.2]
pocon_expert_4tasks_ftOP = [80.84, 66.08, 74.76, 76.84]
pocon_expert_4tasks_copyOP = [81.0, 68.60000000000001, 78.0, 78.36]
pocon_expert_4tasks_d2eOP = [80.52, 68.0, 76.72, 76.8]


# Platicity - 4 tasks all methods
pocon_main_4tasks_scratchOP = []
pocon_main_4tasks_ftOP = [81.84, 71.24000000000001, 79.12, 76.55]
pocon_main_4tasks_ftOP_1500 = [89.3, 91.9, 86.1, 81.5]
pocon_main_4tasks_copyOP = [83.36, 71.08, 79.2, 76.28]
pocon_main_4tasks_d2eOP = [82.04, 70.6, 79.16, 77.44]

plt.plot(pocon_expert_4tasks_scratchOP, 'b-', label='scratchOP')
plt.plot(pocon_expert_4tasks_ftOP, 'c-', label='ftOP - exp')
plt.plot(pocon_expert_4tasks_ftOP_1500, 'c-.', label='ftOP - exp 1500')
plt.plot(pocon_main_4tasks_ftOP, 'c--', label='ftOP - main')
plt.plot(pocon_main_4tasks_ftOP_1500, 'c:', label='ftOP - main 1500')
plt.plot(pocon_expert_4tasks_copyOP, 'g-', label='copyOP - exp')
plt.plot(pocon_main_4tasks_copyOP, 'g--', label='copyOP - exp')
plt.plot(pocon_expert_4tasks_d2eOP, 'r-', label='d2eOP - exp')
plt.plot(pocon_main_4tasks_d2eOP, 'r--', label='d2eOP - main')
plt.title('Last task performance (plasticity) - 4 tasks')
plt.grid()
# plt.legend()


# # #############################################################
# # ##################### 10 tasks ###############################

# # Platicity - 4 tasks all methods
# pocon_expert_4tasks_scratchOP = []
# pocon_expert_4tasks_ftOP = [82.8, 86.5, 81.10, 79.5, 77.0, 82.19, 81.0, 76.1, 83.8, 83.2]
# pocon_expert_4tasks_copyOP = []
# pocon_expert_4tasks_d2eOP = []

# # Platicity - 4 tasks all methods
# pocon_main_4tasks_scratchOP = []
# pocon_main_4tasks_ftOP = [83.0, 87.6, 83.5, 81.11, 78.8, 87.2, 84.2, 80.4, 86.9, 86.0]
# pocon_main_4tasks_copyOP = []
# pocon_main_4tasks_d2eOP = []

# plt.plot(pocon_expert_4tasks_scratchOP, 'b-', label='scratchOP')
# plt.plot(pocon_expert_4tasks_ftOP, 'c-', label='ftOP - exp')
# plt.plot(pocon_main_4tasks_ftOP, 'c--', label='ftOP - main')
# plt.plot(pocon_expert_4tasks_copyOP, 'g-', label='copyOP - exp')
# plt.plot(pocon_main_4tasks_copyOP, 'g--', label='copyOP - exp')
# plt.plot(pocon_expert_4tasks_d2eOP, 'r-', label='d2eOP - exp')
# plt.plot(pocon_main_4tasks_d2eOP, 'r--', label='d2eOP - main')
# plt.title('Last task performance (plasticity) - 10 tasks')
# plt.grid()
# plt.legend()


# [35.18, 37.86, 39.25, 40.129999999999995, 41.23, 41.29, 41.760000000000005, 42.370000000000005, 42.970000000000006, 42.76, 42.83, 44.01, 43.91, 43.47, 44.58, 43.93, 43.66, 43.66, 43.63, 44.31, 42.94, 42.980000000000004, 43.5, 43.54, 43.35, 42.89, 42.89, 43.0, 43.53, 42.730000000000004, 43.72, 43.91, 43.65, 44.12, 43.63, 43.45, 43.72, 43.65, 43.72, 44.379999999999995, 44.87, 44.04, 44.78, 44.26, 44.22, 44.17, 44.21, 44.95, 42.83, 44.21]
