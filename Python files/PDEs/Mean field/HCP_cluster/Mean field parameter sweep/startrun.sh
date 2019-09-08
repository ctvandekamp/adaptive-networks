#!/bin/bash

# This Shell script runs the mean field parameter sweep on the HPC cluster. 
# Using a .txt file containing the parameters and the scipt which is able to solve the model.
# Change the directories to your own!

#PBS -l nodes=1:ppn=1
#PBS -N mean field model parameter sweep
#PBS -o /home/ctvandekamp/mean_field/out.$PBS_JOBID
#PBS -e /home/ctvandekamp/mean_field/err.$PBS_JOBID

parameters=$(sed -n -e "${PBS_ARRAYID}p" /home/ctvandekamp/mean_field/parameters.txt)
parameterArray=($parameters)
 
n=${parameterArray[0]}
m=${parameterArray[1]}
meandegree=${parameterArray[2]}

python  /home/ctvandekamp/mean_field/mean_field_pde.py $n $m $meandegree