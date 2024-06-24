"""
Copyright 2024 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from functools import partial
import json
import multiprocessing
from multiprocessing.pool import Pool
import pandas as pd
from pathlib import Path
import subprocess
from tqdm import tqdm

from mappings import response_mapping
from registration_utils import process_patient


def download_dataset(lumiere_path: Path):
    image_url = 'https://springernature.figshare.com/ndownloader/files/38249697'
    image_destination = lumiere_path / 'imaging.zip'
    command = ['curl', '-L', '-o', image_destination, image_url]
    subprocess.run(command)
    command = ['unzip', '-q', image_destination, '-d', lumiere_path]
    subprocess.run(command)
    image_destination.unlink()
    rating_url = 'https://springernature.figshare.com/ndownloader/files/38369741'
    rating_destination = lumiere_path / 'ratings.csv'
    command = ['curl', '-L', '-o', rating_destination, rating_url]
    subprocess.run(command)


def convert_dataset(lumiere_path: Path, n_pr: int = None):
    if n_pr is None:
        n_pr = min(16, multiprocessing.cpu_count()-2)
    complete_cases = {}
    for patient in sorted((lumiere_path / 'Imaging').iterdir()):
        for visit in sorted(patient.iterdir()):
            if (visit / "HD-GLIO-AUTO-segmentation").exists():
                complete_cases[patient.name] = complete_cases.get(patient.name, []) + [visit.name]

    (lumiere_path / "images").mkdir(exist_ok=True)
    (lumiere_path / "segmentations").mkdir(exist_ok=True)
    (lumiere_path / "images_registered").mkdir(exist_ok=True)
    (lumiere_path / "segmentations_registered").mkdir(exist_ok=True)
    (lumiere_path / "reg_masks").mkdir(exist_ok=True)
    (lumiere_path / "reg_matrices").mkdir(exist_ok=True)
    args = [{patient: visits} for patient, visits in complete_cases.items()]
    with Pool(n_pr) as p:
        registration_meta = list(tqdm(p.imap(partial(process_patient, lumiere_path), args), total=len(args)))
    registration_meta = {k: v for d in registration_meta for k, v in d.items()}

    with open(lumiere_path / 'registration_meta.json', 'w') as f:
        json.dump(registration_meta, f, indent=4)

    rating_df = pd.read_csv(lumiere_path / 'ratings.csv')
    rano_ratings = {}
    grouped_data = rating_df.groupby('Patient')
    for patient, group in grouped_data:
        ratings = group[['Date', 'Rating (according to RANO, PD: Progressive disease, SD: Stable disease, PR: Partial response, CR: Complete response, Pre-Op: Pre-Operative, Post-Op: Post-Operative)']].dropna()
        ratings_dict = dict(zip(ratings['Date'], ratings.iloc[:, 1]))
        rano_ratings[patient] = ratings_dict

    patients_dict = {}
    patient_num = 1
    for patient, ratings in rano_ratings.items():
        patients_list = []
        visits = list(ratings.keys())
        baseline = None
        for visit in visits[1:]:
            if visit not in complete_cases.get(patient, []):
                continue
            rating = ratings[visit].lower().strip()
            if baseline is None:
                if rating in ["pr", "cr", "pre-op", "post-op"]:
                    baseline = visit
                continue
            if rating in ["cr", "pr", "sd", "pd"]:
                patients_list += [{
                    "baseline": f"./images/{patient}_{baseline}",
                    "baseline_seg": f"./segmentations/{patient}_{baseline}.nii.gz",
                    "baseline_registered": f"./images_registered/{patient}_{baseline}_reg",
                    "baseline_seg_registered": f"./segmentations_registered/{patient}_{baseline}_reg.nii.gz",
                    "followup": f"./images/{patient}_{visit}",
                    "followup_seg": f"./segmentations/{patient}_{visit}.nii.gz",
                    "followup_registered": f"./images_registered/{patient}_{visit}_reg",
                    "followup_seg_registered": f"./segmentations_registered/{patient}_{visit}_reg.nii.gz",
                    "response": response_mapping[rating]
                }]
            if rating in ["pr", "cr", "post-op"]:
                baseline = visit
        if patients_list:
            patients_dict[f"patient_{patient_num:03d}"] = {f"case_{i:02d}": case for i, case in enumerate(patients_list, 1)}
            patient_num += 1
    with open(lumiere_path / 'patients.json', 'w') as f:
        json.dump(patients_dict, f, indent=4)


def main():
    parser = argparse.ArgumentParser("Script to download and convert the Lumiere dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to the directory where the dataset should be stored")
    parser.add_argument("--n_pr", type=int, required=False, help="Number of processes used for registration")
    args = parser.parse_args()
    lumiere_path = args.dataset_path / 'Lumiere'
    lumiere_path.mkdir(exist_ok=True, parents=True)
    download_dataset(lumiere_path)
    convert_dataset(lumiere_path, args.n_pr)


if __name__ == "__main__":
    main()