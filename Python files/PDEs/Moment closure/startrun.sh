#!/bin/bash

# Shell scipt runs the parameter sweep for the moment closure approximation model on the cluster
# A .txt file with all parameters and a .py file which solves the model are needed. 
# Change the directories to your own. 

#PBS -l nodes=1:ppn=1,walltime=6:00:00
#PBS -N Long_parametersweep
#PBS -o /home/ctvandekamp/out.$PBS_JOBID
#PBS -e /home/ctvandekamp/err.$PBS_JOBID

parameters=$(sed -n -e "${PBS_ARRAYID}p" /home/ctvandekamp/parameters.txt)
parameterArray=($parameters)
 
eta=${parameterArray[0]}
sigma=${parameterArray[1]}
alpha=${parameterArray[2]}
beta=${parameterArray[3]}

python  /home/ctvandekamp/cont_state_pde_hcp_divorderchange_v2.py $eta $sigma $alpha $beta