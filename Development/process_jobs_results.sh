#!/bin/bash
process_dir=$1
echo $process_dir
echo "mae_euc,mae_ang,mae_geo,rmse_euc,rmse_ang,rmse_geo,LAT,epochs,hidden_dim,batch_size,model,seq_length_input,lr,lr_reducing,lr_epochs,lr_multiplicator,weight_decay" > results/model_tuning_logs/$process_dir.csv

for file in "$process_dir"/*
do
  out=$((tar -xOzf $file job_results/model_parameters_adjust_log.csv) >&1)
  log=$(echo -e "$out" | sed -n '2p')
  echo $log >> results/model_tuning_logs/$process_dir.csv
done
