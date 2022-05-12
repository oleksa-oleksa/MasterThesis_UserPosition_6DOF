#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for n_epochs in $(seq 200 100 1000)
    do
        for batch_size in $(seq 1 1 13)
            do
                for hidden_dim in $(seq 5 10 100)
                    do
                        echo "n_epochs: $n_epochs, batch_size: $((2**$batch_size)), hidden_dim: $hidden_dim"
                    done
            done
    done