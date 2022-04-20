#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=oleksandra.baga@hhi.fraunhofer.de

#SBATCH --job-name=UserPrediction6DOF

# stdout and stderr of this job will go into a file named like the job (%x) with SLURM_JOB_ID (5j)
#SBATCH --output=gpu_outputs/%j_%x.out

#SBATCH --nodes=1

# ask slurm to run at most 1 task (slurm task == OS process) which might have subprocesses/threads
#SBATCH --ntasks=1

# number of cpus/task (threads/subprocesses). 8 is enough. 16 seems a reasonable max. with 4 GPUs on a 72 core machine.
#SBATCH --cpus-per-task=8

# request from the generic resources 1 GPU
#SBATCH --gpus=1

#SBATCH --mem=4G

# Launch the singularity image with --nv for nvidia support.
# The job writes its results to stdout which is directed to the output which starts with the job number file. Check it.

singularity run --nv ./UserPrediction6DOF.sif

