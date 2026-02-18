(Title Pending)
Contributors: Asher T. Wang, Songyeon Lee, Byung-Joon Seung, Karin U. Sorenmo 
Authors: Asher T. Wang, Karin U. Sorenmo, Byung-Joon Seung
Status: Preparing for submission in The Veterinary Journal
Introduction
This repository contains the code files used to generate the results for our paper.
This study introduces a machine learning (ML)-driven diagnostic model leveraging urinary metabolite profiling to classify malignant masses and assist in tumor grading. Using proton nuclear magnetic resonance (NMR) spectroscopy, we analyze 55 tumor-specific urinary metabolites from 156 dogs to develop accurate, non-invasive diagnostic tools.

Included are the analyses of the selected (final) models. Note that figure generation may be omitted for brevity. Both Section A and Section B (leakage free/cross validated) codes and their respective datasets are included. The full repository is available online at https://www.ebi.ac.uk/metabolights/editor/MTBLS2550/.

Software Implementation
The analysis was conducted using Python within Jupyter notebooks.
Key Dependencies
pandas, numpy: Data preprocessing and manipulation
scikit-learn, imbalanced-learn: Machine learning model training and handling class imbalance
matplotlib, seaborn: Data visualization
scipy: Scientific and technical computing
lifelines: Kaplan-Meier survival analysis
jupyterlab: Interactive development environment

Acknowledgements
We acknowledge the support of veterinary oncologists and computational researchers who contributed to the dataset and model validation, as well as professor Zhigen Zhao.
Dataset Reference:
Lee, S., Seung, BJ., Yang, I.S. et al. 1H NMR based urinary metabolites profiling dataset of canine mammary tumors. Sci Data 9, 132 (2022). https://doi.org/10.1038/s41597-022-01229-1
