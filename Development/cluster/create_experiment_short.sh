#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
out=experiment_parameters_short.csv

echo "hidden_dim,batch_size,lr_adam,lr_epochs,lr_multiplicator" > $out

for hidden_dim in $(seq 8 1 10)
    do
        for batch_size in $(seq 8 1 10)
        do
            for lr_adam in $(seq 0.0001 0.0001 0.0001)
            do
                for lr_epochs in $(seq 50 10 50)
                do
                    for lr_multiplicator in $(seq 0.3 0.2 0.5)
                    do
                        echo "$((2**$hidden_dim)),$((2**$batch_size)),$lr_adam,$lr_epochs,$lr_multiplicator" >> $out
                    done
                done
            done
        done
    done
echo "$out is written"
