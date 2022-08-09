#!/bin/bash

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

#SBATCH --mem=16G

# to build container write from ssh terminal
# singularity build --force --fakeroot UserPrediction6DOF.sif UserPrediction6DOF.def

source "/etc/slurm/local_job_dir.sh"
mkdir -p "${LOCAL_JOB_DIR}/job_results"
mkdir -p "${LOCAL_JOB_DIR}/job_results/figures"
mkdir -p "${LOCAL_JOB_DIR}/job_results/tabular"
mkdir -p "${LOCAL_JOB_DIR}/job_results/tabular/distances"
mkdir -p "${LOCAL_JOB_DIR}/job_results/predictions"
mkdir -p "${LOCAL_JOB_DIR}/job_results/losses"
mkdir -p $SLURM_SUBMIT_DIR/jobs_results_lstm1

# run job and bind the output dir
# Launch the singularity image with --nv for nvidia support.
# The job writes its results to stdout which is directed to the output which starts with the job number file. Check it.
singularity run --nv  --bind ${LOCAL_JOB_DIR}:/mnt/output ./UserPrediction6DOF.sif

# Store Intermediate Data and Results Locally
# Doing this after the singularity run call ensures, that the data is copied back even when your singularity run fails.
cd $LOCAL_JOB_DIR
tar -zcvf zz_${SLURM_JOB_ID}.tar job_results
cp zz_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/jobs_results_lstm1
rm -rf ${LOCAL_JOB_DIR}/job_results


