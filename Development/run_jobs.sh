#!/bin/bash

while IFS="," read -r hidden_dim batch_size n_epochs dropout

do
  echo "$hidden_dim $batch_size $n_epochs $dropout"
  export HIDDEN_DIM=$hidden_dim
  export BATCH_SIZE=$batch_size
  export N_EPOCHS=$n_epochs
  export DROPOUT=$dropout
  export RNN_PARAMETERS=1
  echo "env: $HIDDEN_DIM $BATCH_SIZE $N_EPOCHS $DROPOUT"
  nohup sbatch UserPrediction6DOF.sh &
done < <(tail -n +2 $1)
return
