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

import numpy as np
import shutil
import SimpleITK as sitk
import subprocess

from mappings import modalities


def get_mask_from_scans(scans):
    nonzero_mask = np.zeros_like(scans[0]).astype(bool)
    for scan in scans:
        mask = scan!=0
        nonzero_mask = (nonzero_mask | mask)
    return nonzero_mask


def calculate_dice(mask_file_1, mask_file_2):
    mask_1 = sitk.ReadImage(mask_file_1)
    mask_2 = sitk.ReadImage(mask_file_2)
    mask_1_data = sitk.GetArrayFromImage(mask_1)
    mask_2_data = sitk.GetArrayFromImage(mask_2)
    intersection = np.sum(mask_1_data * mask_2_data)
    union = np.sum(mask_1_data) + np.sum(mask_2_data)
    return 2 * intersection / union


def register_visit(lumiere_path, patient, baseline, visit):
    baseline_path = lumiere_path / "Imaging" / patient / baseline / "HD-GLIO-AUTO-segmentation" / "registered"
    visit_path = lumiere_path / "Imaging" / patient / visit / "HD-GLIO-AUTO-segmentation" / "registered"
    images_reg_path = lumiere_path / "images_registered"
    seg_reg_path = lumiere_path / "segmentations_registered"
    masks_path = lumiere_path / "reg_masks"
    matrices_path = lumiere_path / "reg_matrices"

    mat_file_corr = matrices_path / f"mat_{patient}_{visit}_corr.mat"
    mat_file_mut = matrices_path / f"mat_{patient}_{visit}_mut.mat"
    ref_file = baseline_path / f"T1_r2s_bet_reg.nii.gz"
    reg_file = visit_path / f"T1_r2s_bet_reg.nii.gz"
    input_seg_file = visit_path / "segmentation.nii.gz"
    output_seg_file = seg_reg_path / f"{patient}_{visit}_reg.nii.gz"
    mask_file = masks_path / f"mask_{patient}_{visit}.nii.gz"
    mask_file_baseline = masks_path / f"mask_{patient}_{baseline}.nii.gz"
    mask_file_corr = masks_path / f"mask_{patient}_{visit}_corr.nii.gz"
    mask_file_mut = masks_path / f"mask_{patient}_{visit}_mut.nii.gz"
    input_files = [visit_path / f"{mod}_r2s_bet_reg.nii.gz" for mod in modalities]
    output_files = [images_reg_path / f"{patient}_{visit}_reg" / f"{patient}_{visit}_reg_{i:04d}.nii.gz" for i in range(len(modalities))]

    cmd = [
        "flirt", "-in", reg_file, "-ref", ref_file, "-omat", mat_file_corr, "-bins", "256", "-cost", "corratio",
        "-searchrx", "-90", "90", "-searchry", "-90", "90", "-searchrz", "-90", "90", "-dof", "6", "-interp", "spline",
    ]
    subprocess.run(cmd)
    cmd = [
        "flirt", "-in", reg_file, "-ref", ref_file, "-omat", mat_file_mut, "-bins", "256", "-cost", "mutualinfo",
        "-searchrx", "-90", "90", "-searchry", "-90", "90", "-searchrz", "-90", "90", "-dof", "6", "-interp", "spline",
    ]
    subprocess.run(cmd)

    cmd = [
        "flirt", "-in", mask_file, "-ref", ref_file, "-out", mask_file_corr, "-applyxfm", "-init", mat_file_corr, "-interp", "nearestneighbour",
    ]
    subprocess.run(cmd)
    cmd = [
        "flirt", "-in", mask_file, "-ref", ref_file, "-out", mask_file_mut, "-applyxfm", "-init", mat_file_mut, "-interp", "nearestneighbour",
    ]
    subprocess.run(cmd)

    corr_dice = calculate_dice(mask_file_corr, mask_file_baseline)
    mut_dice = calculate_dice(mask_file_mut, mask_file_baseline)
    mat_file = mat_file_corr if corr_dice > mut_dice else mat_file_mut
    mask_file = mask_file_corr if corr_dice > mut_dice else mask_file_mut

    cmd = [
        "flirt", "-in", input_seg_file, "-ref", ref_file, "-out", output_seg_file, "-applyxfm", "-init", mat_file, "-interp", "nearestneighbour",
    ]
    subprocess.run(cmd)

    for input_file, output_file in zip(input_files, output_files):
        cmd = [
            "flirt", "-in", input_file, "-ref", ref_file, "-out", output_file, "-applyxfm", "-init", mat_file, "-interp", "spline",
        ]
        subprocess.run(cmd)
        cmd = [
            "fslmaths", output_file, "-mas", mask_file, output_file
        ]
        subprocess.run(cmd)

    return {"mat_file": str(mat_file), "mask_file": str(mask_file), "corr_dice": corr_dice, "mut_dice": mut_dice}


def process_patient(lumiere_path, patient_dict):
    patient = list(patient_dict.keys())[0]
    visits = list(patient_dict.values())[0]
    images_path = lumiere_path / "images"
    images_reg_path = lumiere_path / "images_registered"
    seg_path = lumiere_path / "segmentations"
    seg_reg_path = lumiere_path / "segmentations_registered"
    masks_path = lumiere_path / "reg_masks"
    registration_meta = {patient: {}}
    for i, visit in enumerate(visits):
        data_path = lumiere_path / "Imaging" / patient / visit / "HD-GLIO-AUTO-segmentation" / "registered"
        (images_path / f"{patient}_{visit}").mkdir(exist_ok=True, parents=True)
        (images_reg_path / f"{patient}_{visit}_reg").mkdir(exist_ok=True, parents=True)
        for j, mod in enumerate(modalities):
            if not (images_path / f"{patient}_{visit}" / f"{patient}_{visit}_{j:04d}.nii.gz").exists():
                shutil.copy(data_path / f"{mod}_r2s_bet_reg.nii.gz", images_path / f"{patient}_{visit}" / f"{patient}_{visit}_{j:04d}.nii.gz")
        if not (seg_path / f"{patient}_{visit}.nii.gz").exists():
            shutil.copy(data_path / "segmentation.nii.gz", seg_path / f"{patient}_{visit}.nii.gz")
        scans_sitk = [sitk.ReadImage(data_path / f"{mod}_r2s_bet_reg.nii.gz") for mod in modalities]
        scans_data = [sitk.GetArrayFromImage(scan) for scan in scans_sitk]
        mask = get_mask_from_scans(scans_data)
        mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask_sitk.CopyInformation(scans_sitk[0])
        sitk.WriteImage(mask_sitk, masks_path / f"mask_{patient}_{visit}.nii.gz")
        if i == 0:
            for j, mod in enumerate(modalities):
                if not (images_reg_path / f"{patient}_{visit}_reg" / f"{patient}_{visit}_reg_{j:04d}.nii.gz").exists():
                    shutil.copy(data_path / f"{mod}_r2s_bet_reg.nii.gz", images_reg_path / f"{patient}_{visit}_reg" / f"{patient}_{visit}_reg_{j:04d}.nii.gz")
            if not (seg_reg_path / f"{patient}_{visit}_reg.nii.gz").exists():
                shutil.copy(data_path / "segmentation.nii.gz", seg_reg_path / f"{patient}_{visit}_reg.nii.gz")
        else:
            meta = register_visit(lumiere_path, patient, visits[0], visit)
            registration_meta[patient][visit] = meta
    return registration_meta