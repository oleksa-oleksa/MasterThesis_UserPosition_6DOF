#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
out=experiment_parameters.csv

echo "hidden_dim,batch_size,lr_adam,lr_epochs,lr_multiplicator,n_epochs" > $out
lr_adam=0.0001
lr_multiplicator=0.5
lr_epochs=0.5
n_epochs=500

for hidden_dim in $(seq 4 1 11)
    do
        for batch_size in $(seq 4 1 11)
        do
           echo "$((2**$hidden_dim)),$((2**$batch_size)),$lr_adam,$lr_epochs,$lr_multiplicator,$n_epochs" >> $out
        done
    done
echo "$out is written"
