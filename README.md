_Copyright © German Cancer Research Center (DKFZ) and contributors. Please make sure that your usage of this code is in compliance with its license:_
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---
# BraTPRO Challenge (MICCAI 2025)

This repository contains the code related to Task 11 - Predicting the Tumor Response During Therapy - of the BraTS-Lighthouse 2025 Challenge.
Also checkout the challenge [Website](https://www.synapse.org/Synapse:syn64153130/wiki/631459)

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

### Submission

Algorithms need to be submitted via docker, with the following script implemented:
```
/workspace/run_inference.sh test_data_dir pred_dir
```
Submitted docker images will then be executed via the following docker command:
```
docker run --gpus all -v "test_data_dir:/mnt/test_data" -v "pred_dir:/mnt/pred" --read-only docker-image-name /workspace/run_inference.sh /mnt/test_data /mnt/pred
```

In order to submit your docker image to the challenge, you will first need to create a project on synapse and give it a meaningful name (this is not required but advised). This project will have a unique Project ID (e.g. syn12345678), which you will need in the following.
After creating your docker image locally, you can upload it to the synapse docker registry

```
docker login docker.synapse.org
docker tag docker_imagename docker.synapse.org/synapse_project_ID/docker_imagename
docker push docker.synapse.org/synapse_project_ID/docker_imagename:latest
```

In order to submit the uploaded Docker image you need to:

- go to your project -> Docker and choose the Docker image you want to submit
- click on Docker Repository Tools -> Submit Docker Repository to Challenge
- choose the correct tag and the evaluation queue you want to submit to