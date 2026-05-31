# Example usage: python evaluation_script.py --dataset ./data/evaluations.csv --prediction_dir ./predictions --algo ours_rc

import os
import json
import numpy as np
import pandas as pd

import argparse
import scripts.helpers as hlp
from scripts.scoring import compute_kendall_stats


def score_performance(pred_file, references, valid_papers, valid_reviewers, bootstraps):
    """Compute the main metric for predicted similarity together with bootstrapped values for confidence intervals

    :param pred_file: Name of the file where predicted similarities are stored (file must be in the PRED_PATH dir)
    :param references: Ground truth values of expertise
    :param valid_papers: Papers to include in evaluation
    :param valid_reviewers: Reviewers to include in evaluation
    :param bootstraps: Subsampled reviewer pools for bootstrap computations
    :return: Score of the predictions + data to compute confidence intervals (if `bootstraps` is not None)
    """

    with open(pred_file, 'r') as handler:
        predictions = json.load(handler)

    stats = compute_kendall_stats(predictions, references, valid_papers, valid_reviewers)
    variations = [compute_kendall_stats(predictions, references, valid_papers, vr) for vr in bootstraps]

    return (
        stats["loss"],
        stats["acc"],
        [v["loss"] for v in variations],
        [v["acc"] for v in variations],
    )

def parse_args(argv=None):
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, help='Path to the dataset csv file')
    parse.add_argument('--prediction_dir', type=str, help='Directory where predictions are stored')
    parse.add_argument('--algo', type=str, help='Name of the algorithm under evaluation')

    return parse.parse_args(argv)


if __name__ == '__main__':

    args = parse_args()

    df = pd.read_csv(args.dataset, sep='\t')
    references, _ = hlp.to_dicts(df)

    all_reviewers = list(references.keys())

    all_papers = set()
    for rev in references:
        all_papers = all_papers.union(references[rev].keys())

    # Prepare reviewer pools for computing Confidence Intervals (n=1,000 iterations)
    bootstraps = [np.random.choice(all_reviewers, len(all_reviewers), replace=True) for x in range(1000)]

    # Check that prediction files are available
    available_predictions = set(os.listdir(args.prediction_dir))

    for iteration in range(1, 11):
        f_name = f"{args.algo}_d_20_{iteration}_ta.json"
        if f_name not in available_predictions:
            raise ValueError(f"Predicted similarities are missing: {f_name}")

    results = {"pointwise_loss": [], "variations_loss": [], "pointwise_acc": [], "variations_acc": []}

    for iteration in range(1, 11):
        f_name = f"{args.algo}_d_20_{iteration}_ta.json"
        f_path = os.path.join(args.prediction_dir, f_name)
        tmp = score_performance(f_path, references, all_papers, all_reviewers, bootstraps)

        results["pointwise_loss"].append(tmp[0])
        results["pointwise_acc"].append(tmp[1])
        results["variations_loss"].append(tmp[2])
        results["variations_acc"].append(tmp[3])

    # Get pointwise estimate of performance
    point_loss = float(np.mean(results["pointwise_loss"]))
    point_acc = float(np.mean(results["pointwise_acc"]))

    # Get 95% confidence interval
    boot_loss = np.matrix(results["variations_loss"]).mean(axis=0).tolist()[0]
    ci_loss = f"[{np.percentile(boot_loss, 2.5):.4f}; {np.percentile(boot_loss, 97.5):.4f}]"
    boot_acc = np.matrix(results["variations_acc"]).mean(axis=0).tolist()[0]
    ci_acc = f"[{np.percentile(boot_acc, 2.5):.4f}; {np.percentile(boot_acc, 97.5):.4f}]"

    print(f"Pointwise estimate of loss: {point_loss:.4f}")
    print(f"95% confidence interval (loss): {ci_loss}")
    print(f"Pointwise estimate of acc: {point_acc:.4f}")
    print(f"95% confidence interval (acc): {ci_acc}")
