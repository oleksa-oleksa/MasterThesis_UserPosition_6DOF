#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
out=experiment_lr.csv

echo "hidden_dim,batch_size,lr_adam,lr_epochs,lr_multiplicator,weight_decay_adam,n_epochs" > $out
lr_epochs=50
n_epochs=500
weight_decay_adam=0.000000000001

for lr_adam in $(seq 0.0001 0.0001 0.001)
    do
        for lr_multiplicator in $(seq 0.3 0.2 0.8)
        do
           echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,$weight_decay_adam,$n_epochs" >> $out
           echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,$weight_decay_adam,$n_epochs" >> $out

        done
    done

for lr_adam in $(seq 0.001 0.002 0.009)
    do
        for lr_multiplicator in $(seq 0.3 0.2 0.8)
        do
           echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,$weight_decay_adam,$n_epochs" >> $out
           echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,$weight_decay_adam,$n_epochs" >> $out

        done
    done

lr_adam=0.0001
lr_multiplicator=0.5

echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.0000000000000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.0000000000000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.000000000000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.000000000000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.00000000000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.00000000000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.0000000000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.0000000000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.000000000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.000000000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.00000000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.00000000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.0000000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.0000000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.000000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.000000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.00000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.00000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.0000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.0000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.000001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.000001,$n_epochs" >> $out
echo "1024,512,$lr_adam,$lr_epochs,$lr_multiplicator,0.00001,$n_epochs" >> $out
echo "2048,1024,$lr_adam,$lr_epochs,$lr_multiplicator,0.00001,$n_epochs" >> $out

echo "$out is written"
