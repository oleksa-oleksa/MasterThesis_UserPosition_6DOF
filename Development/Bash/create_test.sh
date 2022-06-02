#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
out=test_parameters.csv

echo "hidden_dim,batch_size,n_epochs,dropout,layers" > $out

for ((hidden_dim = 10; hidden_dim <= 80; hidden_dim+=5));
do
    for batch_size in $(seq 10 1 13)
    do
        echo "$hidden_dim,$((2**$batch_size)),500,0,1" >> $out
    done
    hidden_dim=$((hidden_dim+hid_step))
done

echo "$out is written"
