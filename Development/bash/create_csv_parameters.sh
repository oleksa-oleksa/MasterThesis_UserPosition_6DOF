#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
hid_step=8
end=450

echo "hidden_dim, batch_size, n_epochs, dropout" > jobs_parameters.csv

for ((hidden_dim = 0; hidden_dim <= end; hid_step));
do
    hidden_dim=$((hidden_dim+hid_step))

    if ((hidden_dim>=24 && hidden_dim<=48));
    then
        hid_step=12
    fi

    if ((hidden_dim==60));
    then
        hidden_dim=100
        hid_step=200
    fi
    for batch_size in $(seq 4 1 13)
    do
        for n_epochs in $(seq 500 250 1000)
        do
            for dropout in $(seq 0.2 0.2 0.6)
            do
                echo "$hidden_dim, $((2**$batch_size)), $n_epochs, $dropout" >> jobs_parameters.csv
            done
        done
    done
done
echo "jobs_parameters.csv is written"
