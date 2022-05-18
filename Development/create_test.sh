#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
out=test_parameters.csv
# dropout=0, layer=1

echo "hidden_dim,batch_size,n_epochs,dropout,layers" > $out

for ((hidden_dim = 200; hidden_dim <= 600; hidden_dim+=200));
do
    for batch_size in $(seq 10 1 13)
    do
        for n_epochs in $(seq 800 100 1000)
        do
            echo "$hidden_dim,$((2**$batch_size)),$n_epochs,0,1" >> $out
        done
    done
    hidden_dim=$((hidden_dim+hid_step))
done

echo "$out is written"
