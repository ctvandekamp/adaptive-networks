'''
Creating a .txt file for a parameter sweep which is compatible with the HPC cluster
'''

import numpy as np

file = open("C:/Users/Carsten/Documents/HCP_cluster/parameters.txt", "w")

n = np.linspace(0, 0.2, 201)
m = [1]
k = [1]
num_decimals = 5

for e in range(len(n)):
    for s in range(len(m)):
        for a in range(len(k)):
            # Rounding to prevent strange floats which differ in the 10th decimal
            n[e] = np.around(n[e], num_decimals)
            m[s] = np.around(m[s], num_decimals)
            k[a] = np.around(k[a], num_decimals)

            file.write((str(n[e]) + ' ' + str(m[s]) + ' ' +
                        str(k[a]) + '\n'))

file.close()
