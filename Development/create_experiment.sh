#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
hid_step=200
end=800

echo "hidden_dim, batch_size, n_epochs, dropout" > experiment_parameters.csv

for ((hidden_dim = 0; hidden_dim <= end; hid_step));
do
    hidden_dim=$((hidden_dim+hid_step))

    for batch_size in $(seq 10 1 13)
    do
        for n_epochs in $(seq 700 200 1100)
        do
            for dropout in $(seq 0.2 0.1 0.6)
            do
                echo "$hidden_dim, $((2**$batch_size)), $n_epochs, $dropout" >> experiment_parameters.csv
            done
        done
    done
done
echo "experiment_parameters.csv is written"
