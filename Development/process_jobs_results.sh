#!/bin/bash
process_dir=gpu_jobs_results

echo "MSE_pos, MSE_rot, RMSE_pos, RMSE_rot, LAT, hidden_size, epochs, batch_size, dropout" > results/model_parameters_merge.csv

for file in "$process_dir"/*
do
  out=$((tar -xOzf $file job_results/model_parameters_adjust_log.csv) 2>&1)

  while IFS="," read -r MSE_pos MSE_rot RMSE_pos RMSE_rot LAT hidden_size epochs batch_size dropout
  do
      echo "$MSE_pos, $MSE_rot, $RMSE_pos, $RMSE_rot, $LAT, $hidden_size, $epochs, $batch_size, $dropout" >> results/model_parameters_merge.csv
  done < <(tail -n +2 "$out")
done
#tar xf gpu_jobs_results/zz_533333.tar job_results/model_parameters_adjust_log.csv