#!/bin/bash
process_dir=gpu_jobs_results_interpolated

echo "MSE_pos,MSE_rot,RMSE_pos,RMSE_rot,LAT,hidden_size,epochs,batch_size,dropout" > results/model_parameters_merge_interpolated_max.csv

for file in "$process_dir"/*
do
  out=$((tar -xOzf $file job_results/model_parameters_adjust_log.csv) >&1)
  log=$(echo -e "$out" | sed -n '2p')
  echo $log >> results/model_parameters_merge_interpolated_max.csv
done
