#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
out=full_set.csv

echo "hidden_dim,batch_size,n_epochs,dropout,layers" > $out

for ((hidden_dim = 20; hidden_dim <= 100; hidden_dim+=10));
do
    for batch_size in $(seq 4 1 13)
    do
        for n_epochs in $(seq 250 250 500)
        do
            for dropout in $(seq 0.1 0.1 0.3)
            do
                for layers in $(seq 1 1 3)
                do
                     echo "$hidden_dim,$((2**$batch_size)),$n_epochs,$dropout,$layers" >> $out
                done
            done
        done
    done
    hidden_dim=$((hidden_dim+hid_step))
done

echo "$out is written"
