#!/bin/bash

while IFS="," read -r hidden_dim batch_size n_epochs dropout

do
  echo "$hidden_dim $batch_size $n_epochs $dropout"
  export HIDDEN_DIM=$hidden_dim
  export BATCH_SIZE=$batch_size
  export N_ECPOCHS=$n_epochs
  export DROPOUT=$dropout
  echo "env: $HIDDEN_DIM $BATCH_SIZE $N_ECPOCHS $DROPOUT"
done < <(tail -n +2 $1)

