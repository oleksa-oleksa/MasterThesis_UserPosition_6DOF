#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=thorsten.selinger@hhi.fraunhofer.de
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=4G

exit_code=0

# include the definition of the LOCAL_JOB_DIR which is autoremoved after each job
source "/etc/slurm/local_job_dir.sh"

# copy the compressed dataset to $LOCAL_JOB_DIR which is created for each individual job locally on the node where the job or steps of it are running
cp ${SLURM_SUBMIT_DIR}/cifar-10-python.tar.gz ${LOCAL_JOB_DIR}
ret_val=$?; if (( $ret_val > $exit_code )); then exit_code=$ret_val; fi

# create the directory dataset before uncompressing into it
mkdir -p ${LOCAL_JOB_DIR}/datasets
ret_val=$?; if (( $ret_val > $exit_code )); then exit_code=$ret_val; fi

# uncompress the dataset on the node where this script is running
tar xvzf ${LOCAL_JOB_DIR}/cifar-10-python.tar.gz -C ${LOCAL_JOB_DIR}/datasets
ret_val=$?; if (( $ret_val > $exit_code )); then exit_code=$ret_val; fi

# launch the singularity image and bind $LOCAL_JOB_DIR on this node to /mnt/datasets as used within the singularity image
srun singularity run --nv --bind ${LOCAL_JOB_DIR}:/mnt/datasets ./lenet5_single_gpu_dataset.sif
ret_val=$?; if (( $ret_val > $exit_code )); then exit_code=$ret_val; fi

exit $exit_code