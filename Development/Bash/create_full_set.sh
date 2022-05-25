#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
out=full_set.csv

echo "hidden_dim,batch_size,n_epochs,dropout,layers" > $out

for ((hidden_dim = 8; hidden_dim <= 128; hidden_dim+=8));
do
    for batch_size in $(seq 4 1 13)
    do
        for n_epochs in $(seq 100 200 1000)
        do
            echo "$hidden_dim,$((2**$batch_size)),$n_epochs,0,1" >> $out
        done
    done
    hidden_dim=$((hidden_dim+hid_step))
done

for dropout in $(seq 0.1 0.05 0.3)
do
    for layers in $(seq 2 1 4)
    do
        echo "100,2048,800,$dropout,$layers" >> $out
        echo "500,2048,800,$dropout,$layers" >> $out
        echo "500,4096,800,$dropout,$layers" >> $out
        echo "500,8192,800,$dropout,$layers" >> $out
    done
done

echo "$out is written"
