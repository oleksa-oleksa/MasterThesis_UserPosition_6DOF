#!/bin/bash
process_dir=$1
echo $process_dir
echo "MAE_pos,MAE_rot,RMSE_pos,RMSE_rot,LAT,hidden_dim,epochs,batch_size,dropout,layers,model,seq_length_input,lr,lr_reducing,weight_decay,lr_epochs,patience,delta,lr_multiplicator" > results/model_tuning_logs/$process_dir.csv

for file in "$process_dir"/*
do
  out=$((tar -xOzf $file job_results/model_parameters_adjust_log.csv) >&1)
  log=$(echo -e "$out" | sed -n '2p')
  echo $log >> results/model_tuning_logs/$process_dir.csv
done
