_Copyright © German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---
# BraTPRO Challenge (MICCAI 2024)

This repository contains the code related to our MICCAI 2024 Brain Tumor Progression Challenge (BraTPRO).
Also checkout the challenge [Website](https://www.synapse.org/bratpro)

### Requirements

Please install docker for submissions to the challenge: https://www.docker.com/get-started

All python requirements can be installed via
```
pip install -r requirements.txt
```
In addition, the temporal registration of images is done using [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/), which can be [installed](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/linux) separately.

### Download the dataset

The python script `dataset_download/download_and_convert_dataset.py` can be used to automatically download and convert the pulbic [LUMIERE Dataset](https://springernature.figshare.com/collections/The_LUMIERE_Dataset_Longitudinal_Glioblastoma_MRI_with_Expert_RANO_Evaluation/5904905) in the suggested dataset format.
Run it using
```
python dataset_download/download_and_convert_dataset.py dataset_location
```
where `dataset_location` is the path where the dataset should be saved.<br>
**Note that this requires at least ~65GB of free space!**

### Evaluation

Code for the metircs used in the challenge evaluation can be found in `evaluation/metrics.py`