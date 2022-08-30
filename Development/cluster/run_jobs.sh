#!/bin/bash

while IFS="," read -r hidden_dim batch_size lr_adam lr_epochs lr_multiplicator n_epochs


do
  echo "$hidden_dim $batch_size $lr_adam $lr_epochs $lr_multiplicator $n_epochs"
  export HIDDEN_DIM=$hidden_dim
  export BATCH_SIZE=$batch_size
  export LR_ADAM=$lr_adam
  export LR_EPOCHS=$lr_epochs
  export LR_MULTIPLICATOR=$lr_multiplicator
  export N_EPOCHS=$n_epochs
  export RNN_PARAMETERS=1
  nohup sbatch UserPrediction6DOF.sh &
done < <(tail -n +2 $1)
