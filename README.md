<h2> Motivation for the research </h2>
The correct and fast head movement prediction is a key to provide a smooth and com- fortable user experience in VR environment during head-mounted display (HDM) usage. The recent improvements in computer graphics, connectivity and the com- putational power of mobile devices simplified the progress in Virtual Reality (VR) technology.

Rendering of volumetric content remains very demanding task for existing devices. Thus the improvement of a perfor- mance of existing methods, design and implementation of new approaches specially for the 6-DoF environment could be a promising research topic.

This research focuses on reducing the Motion-to-Photon (M2P) latency by predicting the future user position and orientation for a look-ahead time (LAT) and sending the corresponding rendered view to a client. The LAT in this approach must be equal or larger to the M2P latency of the network including round-trip time (RTT) and time need for calculation and rendering of a future picture at remote server.

<h2>Installation and Running</h2>

Download the code and create a virtualenv and activate it:


``python3 -m venv UserPrediction6DOF``

``source UserPrediction6DOF/bin/activate``

Install the packages according to the configuration file requirements.txt.

``pip install -r requirements.txt``

All commands must be executed from ``Development`` folder and ``UserPrediction6DOF``folder must be included in it.

Resamples the collected raw traces from HMD  with a sampling time of 5ms and saves them to ./data/interpolated.

``python -m UserPrediction6DOF prepare``

Quaternions between neighboring points in obtained dataset represent the very similar orientation. But on plots can be seen that there are sharp changes of line from negative to positive area with the same amplitude. Thus the two neighboring quaternions with similar rotation have significant 4D vector space between them. It makes prediction worse what can be proved by RMSE and MSE rotation metrics. 
By default the dataset from ./data/interpolated. will be used and result will be saved to ./data/flipped. With -i can be set the input path and with -o output.
 
``python -m UserPrediction6DOF flip``

or

``python -m UserPrediction6DOF flip -i ./data/interpolated -o ./data/flipped``

The dataset with flipped negative quaternions must be normalized

The full normalized dataset uses Min-max normalization that is one of the most common ways to normalize data. For every feature, the minimum value of that feature gets transformed into a 0, the maximum value gets transformed into a 1, and every other value gets transformed into a decimal between 0 and 1.
The result will be saved onto ``./data/normalized_full`` (the ending ``full`` be added by the application)

``python -m UserPrediction6DOF normalize -t full -i ./data/flipped -o ./data/normalized``

Only position data (x, y, z) can be normalized with Min-max normalization:

``python -m UserPrediction6DOF normalize -t min-max -i ./data/flipped -o ./data/normalized``


The position data can be normalized also with z-score normalization. To normalize using standardization, the every value inside the dataset will be transformed to its corresponding value using the formula x = (x - mean) / std:

``python -m UserPrediction6DOF normalize -t mean -i ./data/flipped -o ./data/normalized``


Interpolated, flipped and normalized dataset to train LSTM and GRU Models and to make prediction.
The best resuls are achieved with fully normalized dataset.

Application works with environment variables. If ``RNN_PARAMETERS`` is set (is equal to 1 or something else) then the following parameter must be set as environment variables: ``HIDDEN_DIM, BATCH_SIZE, N_EPOCHS, DROPOUT, LAYERS``.

There are several model implementation. Best are LSTM1 and LSTM3. To run LSTM1 on full interpolated dataset with flipped negative quaternions for LAT 100ms (-w 100) use command: 

``python -m UserPrediction6DOF run -a rnn -m lstm1 -t full -w 100 -d data/flipped '``

Best GRU is GRU1. To run GRU for LAT 100ms use command: 

``python -m UserPrediction6DOF run -a rnn -m gru1 -t full -w 100 -d data/flipped '``

You can check all options with ``python -m UserPrediction6DOF run -h``

<h2>Create train paramaters</h2>

Different train parameters can be used on the step of hyperparameter search. Use the provided Bash-scripts for creating the full test with more then 1000 jobs, experiment with only needed parameters and test set for testing purposes:

``./create_full_set.sh ``
 
``./create_experiment.sh``

``./create_full_set.sh ``


<h2>Slurm and Singularity container</h2>

Training can be done in container in Singularity. 

Modify ``UserPrediction6DOF.sh`` to specify the output directory

``mkdir -p $SLURM_SUBMIT_DIR/gpu_jobs_results_fullNorm_lstm_MSE``

and

``cp zz_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/gpu_jobs_results_fullNorm_lstm_MSE``

Modify ``UserPrediction6DOF.def`` to specify the model and dataset. For example to run LSTM with fully normalized dataset uncomment the line ``    python -m UserPrediction6DOF run -a rnn -m lstm -w 100 -d 'data/normalized_full'``. Comment the rest of lines starting with ``python``.

Build container in Singularity:

``singularity build --force --fakeroot UserPrediction6DOF.sif UserPrediction6DOF.def``


Run set of jobs. Script takes row by row the hyperparameters from specified csv-file, sets environment variables and starts new job.

**Warning!** This will run a bunch of jobs. ``full_set.csv`` runs >1000 jobs! Check your permissions in advance and contact your Slurm/Singularity administrator to ask how many jobs you can run at once. 

To run your model (LSTM or GRU) on selected dataset with full hyperparameters search use command

``./run_jobs.sh Bash/full_set.csv``

To see running (R) and pending (PD) jobs type:

``squeue``

If something went wrong, you can stop all your jobs with the command:

``scancel -u <username>``

<h2>Post-processing job results</h2>

Job results are saved as tar-archive in folder specified in ``UserPrediction6DOF.sh``:

``cp zz_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR/gpu_jobs_results_fullNorm_lstm_MSE``

Run provided Bash-script for opening archives and saving hyperparameters from created by each job ``model_parameters_adjust_log`` into merged csv-file:

``./process_jobs_results.sh gpu_jobs_results_fullNorm_lstm_MSE``

This created merged cvs-file with all hyperparameters in ``./results/gpu_jobs_results_fullNorm_lstm_MSE.csv``.

Use this file for analyze and plotting.

<h2>Analyze of the results</h2>

The results are logged into CSV-file as mentioned before.

<h3>Parameters overview</h3>

To find 10 minimal MAE position and MAE rotation use command:

``python process_results.py min results/gpu_jobs_results_fullNorm_lstm_MSE.csv ``

To find 10 maximal MAE position and MAE rotation use command:

``python process_results.py max results/gpu_jobs_results_fullNorm_lstm_MSE.csv ``


<h2>Plots</h2>

There are many plots available. Plots are saved as pdf-files in ``./results/plotter``

Interpolated dataset: graph with x, y and z position, graph with 4 quaternions parameters and Euler angles in one pdf-file: 

``python -m UserPrediction6DOF plot -p dataset ``

Dataset with flipped quaternions: graph with x, y and z position, graph with 4 quaternions parameters and Euler angles in one pdf-file: 

``python -m UserPrediction6DOF plot -p flipped_quaternions ``

The comparision of interpolated dataset with dataset with flipped negative quaternions. This plots only quaternions graph from both datasets:

``python -m UserPrediction6DOF plot -p compare ``

For some plots the column can be specified with flag -c

Histogram. 

``python -m UserPrediction6DOF plot -p hist -c qw ``

``python -m UserPrediction6DOF plot -p hist -c x``

Correlation matrix:

``python -m UserPrediction6DOF plot -p hist -c qw``

