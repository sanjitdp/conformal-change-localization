import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_1samp, uniform, chi2

from calibrated_confidence import (
    CalibratedFisher,
    CalibratedMinimum,
    CalibratedBonferroni,
)


class ChangepointAnalyzer:

    def __init__(
        self,
        dataset,
        changepoint,
        alpha=0.05,
        test_fn=lambda p: ks_1samp(p, uniform.cdf).pvalue,
        ci_method=CalibratedFisher,
    ):
        self.dataset = dataset
        self.ci = ci_method(self.dataset, test_fn)
        self.alpha = alpha
        self.changepoint = changepoint

    def run_analysis(self):
        self.ci.compute_test()

        self.ci.compute_statistics()

        self.ci.compute_threshold(self.alpha)

        self.ci.compute_confidence_interval()

        return (
            self.ci.statistics,
            self.ci.threshold,
            self.ci.get_ci_endpoints(),
        )

    def get_discrepancy_scores(self):
        x = self.dataset.get_dataset()
        scores_left = self.dataset.get_left_score()
        scores_right = self.dataset.get_right_score()

        length = len(x)
        discrepancy_scores = np.empty(length - 1)
        statistics = []

        rng = np.random.RandomState(42)

        for t in tqdm(range(length - 1), desc="Computing discrepancy scores"):
            p = np.empty(length)

            for r in range(t + 1):
                num_greater = np.sum(scores_left[t, : r + 1] > scores_left[t, r])
                num_equal = np.sum(scores_left[t, : r + 1] == scores_left[t, r])
                p[r] = (num_greater + rng.uniform(0, 1) * num_equal) / (r + 1)

            for r in range(t + 1, length):
                num_greater = np.sum(scores_right[t, r:] > scores_right[t, r])
                num_equal = np.sum(scores_right[t, r:] == scores_right[t, r])
                p[r] = (num_greater + rng.uniform(0, 1) * num_equal) / (length - r)

            left_ks = ks_1samp(p[: t + 1], uniform.cdf)
            right_ks = ks_1samp(p[t + 1 :], uniform.cdf)
            statistics.append((left_ks, right_ks))

            discrepancy_scores[t] = left_ks.statistic * np.sqrt(
                t + 1
            ) + right_ks.statistic * np.sqrt(length - t - 1)

        return discrepancy_scores, statistics

    def plot_discrepancy_scores(self, save_path=None, title=None):
        print("Calculating discrepancy scores...")
        discrepancy_scores, statistics = self.get_discrepancy_scores()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(1, len(discrepancy_scores) + 1), discrepancy_scores)

        if self.changepoint is not None:
            ax.axvline(
                x=self.changepoint,
                color="red",
                linestyle="--",
                label="True Changepoint",
            )

        ax.set_xlabel("Position $t$")
        ax.set_ylabel("Discrepancy Score")

        if title:
            ax.set_title(title)
        else:
            if hasattr(self.dataset, "digit1") and hasattr(self.dataset, "digit2"):
                ax.set_title(
                    f"MNIST Digit Change Detection (Digits {self.dataset.digit1} → {self.dataset.digit2})"
                )
            elif hasattr(self.dataset, "delta"):
                ax.set_title(
                    f"Gaussian Mean Change Detection ($\\delta={self.dataset.delta}$)"
                )
            else:
                ax.set_title("Sentiment Analysis Changepoint Detection")

        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig, statistics

    def plot_p_values(self, save_path=None, title=None):
        if not hasattr(self.ci, "test_p_values"):
            self.ci.compute_test()

        p_values_left = self.ci.test_p_values[:, 0]
        p_values_right = self.ci.test_p_values[:, 1]

        fig, ax = plt.subplots(figsize=(10, 6))

        fisher_stat = -2 * (np.log(p_values_left) + np.log(p_values_right))
        fisher_p = 1 - chi2.cdf(fisher_stat, 4)

        ax.plot(np.arange(1, len(fisher_p) + 1), fisher_p)

        if self.changepoint is not None:
            ax.axvline(
                self.changepoint,
                color="red",
                linestyle="--",
                label=f"Changepoint ($\\xi = {self.changepoint}$)",
            )

        ax.axhline(
            self.alpha,
            color="green",
            linestyle=":",
            label=f"Threshold ($\\alpha = {self.alpha}$)",
        )

        ax.set_xlabel("Position $t$")
        ax.set_ylabel("p-value ($p_t$)")

        if title:
            ax.set_title(title)
        else:
            if hasattr(self.dataset, "digit1") and hasattr(self.dataset, "digit2"):
                ax.set_title(
                    f"p-values for MNIST digit change (Digits {self.dataset.digit1} → {self.dataset.digit2})"
                )
            elif hasattr(self.dataset, "delta"):
                ax.set_title(
                    f"p-values for Gaussian Mean Change ($\\delta={self.dataset.delta}$)"
                )
            else:
                ax.set_title("p-values for Changepoint Detection")

        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        confidence_set = np.argwhere(fisher_p > self.alpha).flatten()
        confidence_interval = (
            (confidence_set[0], confidence_set[-1]) if len(confidence_set) > 0 else None
        )

        print(f"True changepoint: {self.changepoint}")
        print(f"Confidence interval: {confidence_interval}")
        print(f"Maximum Fisher's p-value at t={np.argmax(fisher_p)+1}")

        return fig, fisher_p, confidence_interval

    def plot_basic_results(self, save_path=None, title=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(np.arange(1, len(self.ci.statistics) + 1), self.ci.statistics)

        ax.axhline(
            self.ci.threshold,
            color="red",
            linestyle="--",
            label=f"Threshold ($\\alpha={self.alpha}$)",
        )

        if self.changepoint is not None:
            ax.axvline(
                self.changepoint,
                color="green",
                linestyle=":",
                label=f"True Changepoint ($\\xi={self.changepoint}$)",
            )

        ci_endpoints = self.ci.get_ci_endpoints()
        if ci_endpoints:
            left, right = ci_endpoints
            ax.axvspan(
                left + 1,
                right + 1,
                alpha=0.2,
                color="blue",
                label=f"Confidence Interval: [{left+1}, {right+1}]",
            )

        ax.set_xlabel("Position $t$")
        ax.set_ylabel("Test Statistic")

        if title:
            ax.set_title(title)
        else:
            calibration_mode = self.dataset.calibration_mode
            cal_type = {
                "none": "No Calibration",
                "left": "Left-Side Calibration",
                "right": "Right-Side Calibration",
                "both": "Two-Sided Calibration",
            }[calibration_mode]
            ax.set_title(f"Conformal Changepoint Analysis with {cal_type}")

        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig

    def print_results(self):
        ci_endpoints = self.ci.get_ci_endpoints()

        print("\nChangepoint Analysis Results:")
        print("-" * 40)

        print(f"Dataset length: {self.dataset.length}")
        print(f"True changepoint: {self.changepoint}")

        if hasattr(self.dataset, "digit1") and hasattr(self.dataset, "digit2"):
            print(f"Digits: {self.dataset.digit1} → {self.dataset.digit2}")
        elif hasattr(self.dataset, "delta"):
            print(f"Mean shift magnitude (delta): {self.dataset.delta}")
        elif hasattr(self.dataset, "pre_pos_ratio") and hasattr(
            self.dataset, "post_pos_ratio"
        ):
            print(
                f"Sentiment ratio before: {self.dataset.pre_pos_ratio*100:.1f}% positive"
            )
            print(
                f"Sentiment ratio after: {self.dataset.post_pos_ratio*100:.1f}% positive"
            )

        calibration_mode = self.dataset.calibration_mode
        cal_type = {
            "none": "No Calibration",
            "left": "Left-Side Calibration Only",
            "right": "Right-Side Calibration Only",
            "both": "Two-Sided Calibration",
        }[calibration_mode]
        print(f"Calibration mode: {cal_type}")

        if calibration_mode in ["left", "both"]:
            print(f"Left calibration size: {self.dataset.calibration_size}")
        if calibration_mode in ["right", "both"]:
            print(f"Right calibration size: {self.dataset.calibration_size}")

        ci_method = self.ci.__class__.__name__
        print(f"Confidence interval method: {ci_method}")
        print(f"Significance level: {self.alpha}")

        if ci_endpoints:
            left, right = ci_endpoints
            print(f"Confidence interval: [{left+1}, {right+1}]")
            ci_len = right - left + 1
            print(f"CI length: {ci_len}")

            if self.changepoint is not None:
                if left <= self.changepoint <= right:
                    print("✓ True changepoint is contained in the confidence interval")
                else:
                    print(
                        "✗ True changepoint is NOT contained in the confidence interval"
                    )
        else:
            print("No valid confidence interval found")

        if hasattr(self.ci, "test_p_values"):
            p_values = self.ci.test_p_values
            min_idx = np.argmin(np.max(p_values, axis=1))
            min_pval = max(p_values[min_idx])
            print(f"Position with most significant change: {min_idx+1}")
            print(f"Maximum p-value at this position: {min_pval:.6f}")
