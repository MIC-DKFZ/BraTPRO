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
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


response_mapping = {
    0: "Complete Response",
    1: "Partial Response",
    2: "Stable Disease",
    3: "Progressive Disease"
}


def get_tp_fp_fn_tn(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    tn = np.sum((1 - y_true) * (1 - y_pred))
    return tp, fp, fn, tn


def evaluate(prediction: dict, ground_truth: dict):
    y_true = []
    y_pred_prob = []
    for patient, cases in ground_truth.items():
        for case, meta in cases.items():
            y_true.append(meta["response"])
            try:
                y_pred_prob.append(prediction[patient][case]["response"])
            except KeyError:
                y_pred_prob.append([0., 0., 0., 0.])

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    y_pred = np.argmax(y_pred_prob, axis=1)

    metrics = {}
    
    for cls in range(y_pred_prob.shape[1]):
        cls_true = (y_true == cls).astype(int)
        cls_pred = (y_pred == cls).astype(int)
        cls_pred_prob = y_pred_prob[:, cls]

        tp, fp, fn, tn = get_tp_fp_fn_tn(cls_true, cls_pred)

        metrics[response_mapping[cls]] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'balanced_accuracy': float(1/2 * (tp / (tp + fn) + tn / (tn + fp))),
            'f1_score': float(2 * tp / (2 * tp + fp + fn)),
            'true_positive_rate': float(tp / (tp + fn)),
            'true_negative_rate': float(tn / (tn + fp)),
            'average_precision': float(average_precision_score(cls_true, cls_pred_prob)),
            'roc_auc': float(roc_auc_score(cls_true, cls_pred_prob))
        }

    return metrics


def main():
    parser = argparse.ArgumentParser("Script to evaluate the predictions")
    parser.add_argument("prediction_file", type=Path, help="Path to the file with the predictions")
    parser.add_argument("ground_truth_file", type=Path, help="Path to the file with the ground truth")
    parser.add_argument("output_file", type=Path, help="Path to the output file")
    args = parser.parse_args(("/home/y033f/DataDrive/BraTPRO/test_docker/validation/prediction/prediction.json /home/y033f/DataDrive/BraTPRO/test_docker/validation/training/patients.json /home/y033f/DataDrive/BraTPRO/test_docker/validation/output.json").split())
    with open(args.prediction_file, 'r') as f:
        prediction = json.load(f)
    with open(args.ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    metrics = evaluate(prediction, ground_truth)
    with open(args.output_file, 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()