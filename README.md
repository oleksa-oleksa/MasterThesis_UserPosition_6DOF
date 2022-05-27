<h2> Motivation for the research </h2>
The correct and fast head movement prediction is a key to provide a smooth and com- fortable user experience in VR environment during head-mounted display (HDM) usage. The recent improvements in computer graphics, connectivity and the com- putational power of mobile devices simplified the progress in Virtual Reality (VR) technology.

Rendering of volumetric content remains very demanding task for existing devices. Thus the improvement of a perfor- mance of existing methods, design and implementation of new approaches specially for the 6-DoF environment could be a promising research topic.

This research focuses on reducing the Motion-to-Photon (M2P) latency by predicting the future user position and orientation for a look-ahead time (LAT) and sending the corresponding rendered view to a client. The LAT in this approach must be equal or larger to the M2P latency of the network including round-trip time (RTT) and time need for calculation and rendering of a future picture at remote server.

<h2>Installation and Running</h2>
Download and install the packages according to the configuration file requirements.txt.

``$ pip install -r requirements.txt``

Resamples the collected raw traces from HMD  with a sampling time of 5ms and saves them to ./data/interpolated.

``python -m pred6dof prepare``