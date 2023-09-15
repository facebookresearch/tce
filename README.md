# TCE: A Test-Based Approach to Measuring Calibration Error

## About

Test-Based Calibration Error (TCE) is a new calibration error metric based on statistical hypothesis testing. The value of TCE has a clear interpretation as a percentage of model predictions that deviates significantly from empirical probabilities estimated from data. TCE produces values robust to class imbalance in a normalised, comparable range [0, 100]. TCE is accompanied with a new visual representation that facilitates a better understanding of calibration performance. 

This repository contains sources codes for the following paper:

> [Matsubara T, Tax N, Mudd R, Guy I, TCE: A Test-Based Approach to Measuring Calibration Error.](https://proceedings.mlr.press/v216/matsubara23a.html) *Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence*, PMLR 216:1390-1400, 2023.

The Jupyter notebooks in this repository reproduce the experiments presented in the paper. Implementation of TCE and the visual representation can be found in the *calibration_metric.py* file.



## Dataset

#### UCI Data

One of the experiments uses 9 UCI datasets preprocessed for imbalanced classification benchmark by the *imbalanced-learn* package. They are directly loaded from the package (see https://imbalanced-learn.org/stable/datasets/index.html). For comparison, the experiment additionally uses another UCI dataset *spambase* that has well-balanced prevalance. The *spambase* dataset can be downloaded from https://archive.ics.uci.edu/dataset/94/spambase and placed under the directory "Dataset/UCI/" to be used in the code.

#### ImageNet1000 Data

One of the experiments uses ImageNet1000 data. The ImageNet1000 dataset can be downloaded from https://www.image-net.org/index.php and the *imagenet_root* folder can be placed under the directory "Dataset/DL" to be used in the code. The *generate_label.py* file in the directory is used to preprocess the ImageNet1000 dataset. It generates local files in the directory that contain correct labels and predictive outputs for the dataset to be used in the code.



## License

The majority of TCE is licensed under CC-BY-NC (see LICENSE file), however portions of the project are available under separate license terms: scikit-learn is licensed under the BSD-3-Clause license.
