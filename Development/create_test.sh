#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
hid_step=4
end=4
out=test_parameters.csv

echo "hidden_dim,batch_size,n_epochs,dropout" > $out

for ((hidden_dim = 0; hidden_dim <= end; hid_step));
do
    hidden_dim=$((hidden_dim+hid_step))

    for batch_size in $(seq 4 1 5)
    do
        for n_epochs in $(seq 1 1 3)
        do
            for dropout in $(seq 0 0.1 0.2)
            do
                echo "$hidden_dim, $((2**$batch_size)), $n_epochs, $dropout" >> $out
            done
        done
    done
done
echo "$out is written"
