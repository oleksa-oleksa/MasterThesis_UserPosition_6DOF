#!/bin/bash

while IFS="," read -r hidden_dim batch_size n_epochs dropout layers
do
  echo "$hidden_dim $batch_size $n_epochs $dropout"
  export HIDDEN_DIM=$hidden_dim
  export BATCH_SIZE=$batch_size
  export N_EPOCHS=$n_epochs
  export DROPOUT=$dropout
  export LAYERS=$layers
  export RNN_PARAMETERS=1
  echo "env: $HIDDEN_DIM $BATCH_SIZE $N_EPOCHS $DROPOUT"
  nohup python -m UserPrediction6DOF run -a lstm -w $2 &
done < <(tail -n +2 $1)
