# Compressed Sensing vs Deep Learning for Seismic Full-Waveform Inversion (FWI) on OpenFWI

This project investigates Compressed Sensing (CS) techniques as an interpretable and lightweight alternative to deep learning methods in seismic Full-Waveform Inversion (FWI). Using K-SVD dictionary learning and Orthogonal Matching Pursuit (OMP), we reconstruct seismic velocity maps from the OpenFWI dataset and compare their performance against data-driven models like InversionNet.

While deep learning models generally achieve higher accuracy in complex scenarios, our CS-based pipeline is valuable in data-scarce or resource-limited settings, avoiding the need for end-to-end training and GPU acceleration.

## PROJECT STRUCTURE

OpenFWI_CS_Methods.ipynb: Main Jupyter Notebook implementing the full CS-based inversion pipeline.

OpenFWI_Report.pdf: Comprehensive report explaining methodology, experiments, and results.


## HOW TO USE

Install Dependencies
Install the required Python packages:
pip install numpy scipy scikit-learn matplotlib

### Prepare Data

Download OpenFWI dataset samples (e.g., FlatVel-A, CurveVel-B).

Preprocess seismic data and velocity maps as explained in the notebook.

Typical data shapes:

Velocity models: (B, 1, 70, 70)

Seismic data: (B, 5, 1000, 70)

### Run the Notebook
Open OpenFWI_CS_Methods.ipynb in Jupyter and execute the cells step by step:

Preprocess velocity and seismic data using PCA.

Train the K-SVD dictionary on velocity patches.

Use OMP for patch-based sparse reconstruction.

Visualize reconstruction results and error plots.

## METHOD: COMPRESSED SENSING APPROACH
In this project, we reformulate FWI as a sparse representation problem. The key idea is to assume that local velocity model patches can be efficiently represented as sparse linear combinations of learned dictionary atoms.

Patch Extraction and Preprocessing

Each 2D velocity map is divided into overlapping patches (e.g., 8×8 with stride 4).

Patches are vectorized and normalized.

PCA is applied to reduce dimensionality, highlighting dominant structural variations while mitigating noise.

Dictionary Learning via K-SVD

We jointly learn an overcomplete dictionary D and sparse coefficient vectors alpha for the training patches.

The K-SVD algorithm alternates between:

Sparse coding with OMP to find sparse coefficients.

Dictionary atom updates using singular value decomposition (SVD).

Training stops when the residual error stabilizes or after a set number of iterations.

Sparse Reconstruction with OMP

At inference, each test patch is projected into the PCA-reduced space.

Sparse coefficients alpha are computed by minimizing the reconstruction error with a sparsity constraint.

The reconstructed patch is obtained by mapping back from PCA space to the original space using the learned dictionary.

Reconstructing the Global Velocity Map

Reconstructed patches are combined with overlapping regions averaged (often using Gaussian weighting) to form the final velocity map.

## RESULTS
Experiments were performed on four OpenFWI subsets: FlatVel-A, FlatVel-B, CurveVel-A, and CurveVel-B. Key observations:

On FlatVel-A, K-SVD achieves reasonable reconstructions with errors decreasing as the training set grows.

On CurveVel-B, CS-based recovery struggles due to complex geological structures but still provides physically plausible low-resolution reconstructions.

Comparison with InversionNet (a CNN baseline) shows that deep networks reach 3x lower errors but require GPUs and extensive training data.

For detailed results, including MSE and MAE plots, see the report and the notebook.

## FUTURE WORK

Integrate Total Variation (TV) regularization to better preserve geological boundaries.

Explore plug-and-play priors to combine CS-based edge preservation with learned seismic data features.

Extend experiments to real-world seismic data for evaluating generalization.

## CITATION
If you find this work useful, please cite:
Zhongyu Lin, Zhijun Zhang, Zhehao Zhang. "A Comparative Study of Compressed Sensing and Deep Learning Approaches for Seismic Full-Waveform Inversion on OpenFWI". Johns Hopkins University, 2025.

ACKNOWLEDGMENTS

Professor Trac Tran for invaluable guidance.

OpenFWI dataset by Deng et al.

K-SVD algorithm by Aharon et al. (2006).

CONTACT
For questions or collaboration:

Zhongyu Lin (zhongyu0720@gmail.com)

Zhijun Zhang

Zhehao Zhang
