# SimCMR
## Large-Scale Network Community Detection using Similarity-Guided Merge and Refinement

SimCMR is a network community detection algorithm particularly developed for detecting communities in large-scale networks fast and effectively. It was designed for detecting non-overlapping communities in unipartite, undirected, and unweighted networks.

## Code
SimCMR was implemented in pure Python on top of NetworkX which is a very popular network analysis framework written purely in Python. You can find my own implementation in **code** directory (tunali_community.py). In the same folder, you can find the Python code that I have used to generate LFR benchmark networks as well as the code to run the experiments explained in the published research paper.

## Data
In **data** directory, I share all real-world and artificial benchmark networks in a unified format (edge list with node indices starting from 1). In addition to network files, you can find their corresponding ground-truth community assignments.

## Executables
In **executables** directory, I share all executables that I have used to generate LFR benchmarks and some implementations of other community detection algorithms I have compared SimCMR in my research.

## How to cite
If you find any material here useful, please cite my research as below:

V. Tunali, "Large-Scale Network Community Detection using Similarity-Guided Merge and Refinement", IEEE Access, vol. 9, 2021, doi: 10.1109/ACCESS.2021.3083971

Link to the paper on [IEEE Xplore](https://doi.org/10.1109/ACCESS.2021.3083971)
