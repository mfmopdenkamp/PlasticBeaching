#!/bin/bash -l
#
#SBATCH -J interpol_era5  # the name of your job
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 10:00:00         # time in hh:mm:ss you want to reserve for the job
#SBATCH -n 1                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH -o lorentz_output/output.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e lorentz_output/error.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM

cd /nethome/5586100/PlasticBeaching
conda activate beaching      # this passes your conda environment to all the compute nodes
srun python3 get_wind_era5.py
