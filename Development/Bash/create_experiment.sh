#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
out=experiment_parameters.csv

echo "hidden_dim,batch_size,lr_adam,lr_epochs,lr_multiplicator" > $out

for hidden_dim in $(seq 5 1 8)
    do
        for batch_size in $(seq 7 1 8)
        do
            for lr_adam in $(seq 0.0001 0.0002 0.0006)
            do
                for lr_epochs in $(seq 30 20 70)
                do
                    for lr_multiplicator in $(seq 0.1 0.1 0.5)
                    do
                        echo "$((2**$hidden_dim)),$((2**$batch_size)),$lr_adam,$lr_epochs,$lr_multiplicator" >> $out
                    done
                done
            done
        done
    done
echo "$out is written"
