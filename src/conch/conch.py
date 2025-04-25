import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_1samp, uniform

import matplotlib_inline.backend_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("svg")
try:
    plt.style.use("math.mplstyle")
except:
    print("Warning: math.mplstyle not found, using default style")

from gaussian_dataset import GaussianMeanChangeDataset
from mnist_dataset import MNISTChangepointDataset
from sentiment_dataset import SentimentChangepointDataset
from calibrated_confidence import (
    CalibratedFisher,
    CalibratedMinimum,
    CalibratedBonferroni,
)
from changepoint_analyzer import ChangepointAnalyzer

os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)


def run_gaussian_no_calibration():
    print("\n==== Running Gaussian Mean Change Detection (No Calibration) ====")

    length = 1000
    changepoint = 400
    delta = 0.5
    alpha = 0.05

    print("Generating Gaussian dataset...")
    dataset = GaussianMeanChangeDataset(
        length=length,
        changepoint=changepoint,
        delta=delta,
        calibration_mode="none",
        random_seed=42,
    )

    print("Visualizing dataset...")
    dataset.visualize_samples(save_path="images/gaussian_samples.png")

    print("Analyzing changepoint...")
    analyzer = ChangepointAnalyzer(
        dataset=dataset,
        changepoint=changepoint,
        alpha=alpha,
        test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
        ci_method=CalibratedFisher,
    )

    statistics, threshold, ci_endpoints = analyzer.run_analysis()

    print("Plotting results...")
    analyzer.plot_basic_results(
        save_path="images/gaussian_basic_results.png",
        title=f"Gaussian Mean Change Detection ($\\delta={delta}$)",
    )

    analyzer.print_results()

    return analyzer


def run_gaussian_with_calibration():
    print("\n==== Running Gaussian Mean Change Detection (With Calibration) ====")

    length = 1000
    changepoint = 400
    delta = 0.5
    alpha = 0.05
    calibration_size = 100

    print("Generating Gaussian dataset with calibration...")
    dataset = GaussianMeanChangeDataset(
        length=length,
        changepoint=changepoint,
        delta=delta,
        calibration_mode="both",
        calibration_size=calibration_size,
        random_seed=42,
    )

    print("Analyzing changepoint...")
    analyzer = ChangepointAnalyzer(
        dataset=dataset,
        changepoint=changepoint,
        alpha=alpha,
        test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
        ci_method=CalibratedFisher,
    )

    statistics, threshold, ci_endpoints = analyzer.run_analysis()

    print("Plotting results...")
    analyzer.plot_basic_results(
        save_path="images/gaussian_calibrated_results.png",
        title=f"Gaussian Mean Change Detection with Calibration ($\\delta={delta}$)",
    )

    analyzer.print_results()

    return analyzer


def run_mnist_basic():
    print("\n==== Running MNIST Digit Change Detection ====")

    length = 1000
    changepoint = 400
    alpha = 0.05
    digit1 = 3
    digit2 = 7

    print("Generating MNIST dataset...")
    dataset = MNISTChangepointDataset(
        length=length,
        changepoint=changepoint,
        digit1=digit1,
        digit2=digit2,
        calibration_mode="none",
    )

    print("Visualizing sample images...")
    dataset.visualize_samples(save_path="images/mnist_samples.png")

    print("Analyzing changepoint...")
    analyzer = ChangepointAnalyzer(
        dataset=dataset,
        changepoint=changepoint,
        alpha=alpha,
        test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
        ci_method=CalibratedFisher,
    )

    statistics, threshold, ci_endpoints = analyzer.run_analysis()

    print("Plotting results...")
    analyzer.plot_basic_results(
        save_path="images/mnist_basic_results.png",
        title=f"MNIST Digit Change Detection (Digits {digit1} → {digit2})",
    )

    analyzer.print_results()

    return analyzer


def run_mnist_with_calibration():
    print("\n==== Running MNIST Digit Change Detection (With Calibration) ====")

    length = 1000
    changepoint = 400
    alpha = 0.05
    calibration_size = 100

    print("Generating MNIST dataset with calibration...")
    dataset = MNISTChangepointDataset(
        length=length,
        changepoint=changepoint,
        calibration_mode="both",
        calibration_size=calibration_size,
    )

    print("Analyzing changepoint...")
    analyzer = ChangepointAnalyzer(
        dataset=dataset,
        changepoint=changepoint,
        alpha=alpha,
        test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
        ci_method=CalibratedFisher,
    )

    statistics, threshold, ci_endpoints = analyzer.run_analysis()

    print("Plotting results...")
    analyzer.plot_basic_results(
        save_path="images/mnist_calibrated_results.png",
        title="MNIST Digit Change Detection with Calibration",
    )

    analyzer.print_results()

    return analyzer


def run_sentiment_basic():
    print("\n==== Running SST-2 Sentiment Change Detection ====")

    length = 500
    changepoint = 200
    alpha = 0.05

    print("Generating SST-2 sentiment dataset...")
    dataset = SentimentChangepointDataset(
        length=length,
        changepoint=changepoint,
        mixed_mode=False,
        pre_pos_ratio=1.0,
        post_pos_ratio=0.0,
        calibration_mode="none",
        device="cpu",
    )

    print("Visualizing sample texts...")
    dataset.visualize_samples(save_path="images/sentiment_samples.png")

    print("Analyzing changepoint...")
    analyzer = ChangepointAnalyzer(
        dataset=dataset,
        changepoint=changepoint,
        alpha=alpha,
        test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
        ci_method=CalibratedFisher,
    )

    statistics, threshold, ci_endpoints = analyzer.run_analysis()

    print("Plotting results...")
    analyzer.plot_basic_results(
        save_path="images/sentiment_basic_results.png",
        title="SST-2 Sentiment Change Detection",
    )

    analyzer.print_results()

    return analyzer


def run_sentiment_mixed():
    print("\n==== Running SST-2 Mixed Sentiment Change Detection ====")

    length = 1000
    changepoint = 400
    alpha = 0.05

    print("Generating SST-2 mixed sentiment dataset...")
    dataset = SentimentChangepointDataset(
        length=length,
        changepoint=changepoint,
        mixed_mode=True,
        pre_pos_ratio=0.6,
        post_pos_ratio=0.4,
        calibration_mode="none",
        device="cpu",
    )

    print("Visualizing sample texts...")
    dataset.visualize_samples(save_path="images/sentiment_mixed_samples.png")

    print("Analyzing changepoint...")
    analyzer = ChangepointAnalyzer(
        dataset=dataset,
        changepoint=changepoint,
        alpha=alpha,
        test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
        ci_method=CalibratedFisher,
    )

    statistics, threshold, ci_endpoints = analyzer.run_analysis()

    print("Plotting results...")
    analyzer.plot_basic_results(
        save_path="images/sentiment_mixed_results.png",
        title="SST-2 Mixed Sentiment Change Detection\n(Pre: 60% pos/40% neg, Post: 40% pos/60% neg)",
    )
    analyzer.plot_p_values(
        save_path="images/sentiment_mixed_pvalues.png",
        title="P-values for SST-2 Mixed Sentiment Change",
    )

    analyzer.print_results()

    return analyzer


def run_sentiment_with_calibration():
    print("\n==== Running SST-2 Sentiment Change Detection (With Calibration) ====")

    length = 500
    changepoint = 200
    alpha = 0.05
    calibration_size = 100

    print("Generating SST-2 sentiment dataset with calibration...")
    dataset = SentimentChangepointDataset(
        length=length,
        changepoint=changepoint,
        mixed_mode=False,
        pre_pos_ratio=1.0,
        post_pos_ratio=0.0,
        calibration_mode="both",
        calibration_size=calibration_size,
        device="cpu",
    )

    print("Analyzing changepoint...")
    analyzer = ChangepointAnalyzer(
        dataset=dataset,
        changepoint=changepoint,
        alpha=alpha,
        test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
        ci_method=CalibratedFisher,
    )

    statistics, threshold, ci_endpoints = analyzer.run_analysis()

    print("Plotting results...")
    analyzer.plot_basic_results(
        save_path="images/sentiment_calibrated_results.png",
        title="SST-2 Sentiment Change Detection with Calibration",
    )

    analyzer.print_results()

    return analyzer


def compare_all_datasets():
    print("\n==== Comparing Conformal Changepoint Detection Across Datasets ====")

    alpha = 0.05

    datasets = [
        {
            "name": "Gaussian",
            "object": GaussianMeanChangeDataset(
                length=1000,
                changepoint=400,
                delta=0.5,
                calibration_mode="none",
                random_seed=42,
            ),
            "title": "Gaussian Mean Change",
        },
        {
            "name": "MNIST",
            "object": MNISTChangepointDataset(
                length=1000,
                changepoint=400,
                digit1=3,
                digit2=7,
                calibration_mode="none",
            ),
            "title": "MNIST Digit Change (3→7)",
        },
        {
            "name": "SST-2",
            "object": SentimentChangepointDataset(
                length=500,
                changepoint=200,
                mixed_mode=False,
                pre_pos_ratio=1.0,
                post_pos_ratio=0.0,
                calibration_mode="none",
                device="cpu",
            ),
            "title": "SST-2 Sentiment Change",
        },
    ]

    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 5 * len(datasets)))

    results = {}

    for i, dataset_info in enumerate(datasets):
        name = dataset_info["name"]
        dataset = dataset_info["object"]
        title = dataset_info["title"]
        changepoint = dataset.changepoint

        print(f"\nAnalyzing {name} dataset...")

        analyzer = ChangepointAnalyzer(
            dataset=dataset,
            changepoint=changepoint,
            alpha=alpha,
            test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
            ci_method=CalibratedFisher,
        )

        statistics, threshold, ci_endpoints = analyzer.run_analysis()

        results[name] = {
            "statistics": statistics,
            "threshold": threshold,
            "ci_endpoints": ci_endpoints,
            "analyzer": analyzer,
        }

        ax = axes[i]
        ax.plot(np.arange(1, len(statistics) + 1), statistics)
        ax.axhline(threshold, color="r", linestyle="--", label=f"Threshold (α={alpha})")

        if changepoint is not None:
            ax.axvline(changepoint, color="g", linestyle=":", label="True Changepoint")

        if ci_endpoints:
            left, right = ci_endpoints
            ax.axvspan(
                left + 1,
                right + 1,
                alpha=0.2,
                color="b",
                label=f"CI: [{left+1}, {right+1}]",
            )

        ax.set_ylabel("Statistic")
        ax.set_title(title)
        ax.legend()

        if i == len(datasets) - 1:
            ax.set_xlabel("Position $t$")

    plt.suptitle(
        "Comparison of Conformal Changepoint Detection Across Datasets", fontsize=16
    )
    plt.tight_layout()
    plt.savefig("images/cross_dataset_comparison.png")

    print("\nSummary of Results:")
    print("-" * 70)
    print(
        f"{'Dataset':<10} {'TP':<5} {'CI Start':<10} {'CI End':<10} {'CI Length':<10} {'Contains CP':<15}"
    )
    print("-" * 70)

    for name in [d["name"] for d in datasets]:
        changepoint = datasets[[d["name"] for d in datasets].index(name)][
            "object"
        ].changepoint
        ci = results[name]["ci_endpoints"]
        if ci:
            left, right = ci
            ci_len = right - left + 1
            contains = "Yes" if left <= changepoint <= right else "No"
            print(
                f"{name:<10} {changepoint:<5} {left+1:<10} {right+1:<10} {ci_len:<10} {contains:<15}"
            )
        else:
            print(
                f"{name:<10} {changepoint:<5} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<15}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Conformal Changepoint Detection")
    parser.add_argument(
        "--example",
        choices=[
            "all",
            "gaussian",
            "gaussian_cal",
            "mnist",
            "mnist_cal",
            "sentiment",
            "sentiment_mixed",
            "sentiment_cal",
            "compare",
        ],
        default="all",
        help="Which example to run (default: all)",
    )
    args = parser.parse_args()

    print("Conformal Changepoint Detection")
    print("=" * 60)

    if args.example in ["all", "gaussian"]:
        run_gaussian_no_calibration()

    if args.example in ["all", "gaussian_cal"]:
        run_gaussian_with_calibration()

    if args.example in ["all", "mnist"]:
        run_mnist_basic()

    if args.example in ["all", "mnist_cal"]:
        run_mnist_with_calibration()

    if args.example in ["all", "sentiment"]:
        run_sentiment_basic()

    if args.example in ["all", "sentiment_mixed"]:
        run_sentiment_mixed()

    if args.example in ["all", "sentiment_cal"]:
        run_sentiment_with_calibration()

    if args.example in ["all", "compare"]:
        compare_all_datasets()

    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()
