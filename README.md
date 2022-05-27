<h2> Motivation for the research </h2>
The correct and fast head movement prediction is a key to provide a smooth and com- fortable user experience in VR environment during head-mounted display (HDM) usage. The recent improvements in computer graphics, connectivity and the com- putational power of mobile devices simplified the progress in Virtual Reality (VR) technology.

Rendering of volumetric content remains very demanding task for existing devices. Thus the improvement of a perfor- mance of existing methods, design and implementation of new approaches specially for the 6-DoF environment could be a promising research topic.

This research focuses on reducing the Motion-to-Photon (M2P) latency by predicting the future user position and orientation for a look-ahead time (LAT) and sending the corresponding rendered view to a client. The LAT in this approach must be equal or larger to the M2P latency of the network including round-trip time (RTT) and time need for calculation and rendering of a future picture at remote server.

<h2>Installation and Running</h2>
Download the code and create a virtualenv and activate it:

``python3 -m venv UserPrediction6DOF``

``source pred6dof/bin/activate``

Install the packages according to the configuration file requirements.txt.

``pip install -r requirements.txt``

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

Only position data (x, y, z) can be normalized with Min-max normalization

The position data can be normalized also with z-score normalization. To normalize using standardization, the every value inside the dataset will be transformed to its corresponding value using the formula 

$x=\frac{x-mean}{std}
:


You can check all options with ``python -m UserPrediction6DOF run -h``