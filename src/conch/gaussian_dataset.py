import numpy as np
import matplotlib.pyplot as plt


class GaussianMeanChangeDataset:

    def __init__(
        self,
        length,
        changepoint,
        delta,
        calibration_mode="none",
        calibration_size=100,
        variance=1.0,
        random_seed=None,
    ):
        self.length = length
        self.changepoint = changepoint
        self.delta = delta
        self.calibration_mode = calibration_mode
        self.calibration_size = calibration_size
        self.variance = variance

        self._validate_inputs()

        if random_seed is not None:
            np.random.seed(random_seed)

        if calibration_mode == "none":
            self.x = self.generate_basic_dataset()
            self.x_cal_pre = None
            self.x_cal_post = None
        elif calibration_mode == "left":
            self.x, self.x_cal_pre = self.generate_left_calibration_dataset()
            self.x_cal_post = None
        elif calibration_mode == "right":
            self.x, self.x_cal_post = self.generate_right_calibration_dataset()
            self.x_cal_pre = None
        elif calibration_mode == "both":
            self.x, self.x_cal_pre, self.x_cal_post = (
                self.generate_dual_calibration_dataset()
            )
        else:
            raise ValueError(f"Invalid calibration mode: {calibration_mode}")

        self.compute_scores()

    def _validate_inputs(self):
        if self.length <= 0:
            raise ValueError("Length must be positive")

        if self.changepoint is not None:
            if self.changepoint < 0 or self.changepoint >= self.length - 1:
                raise ValueError(
                    f"Changepoint must be between 0 and {self.length-2} (inclusive)"
                )

        if self.calibration_size <= 0:
            raise ValueError("Calibration size must be positive")

        if self.variance <= 0:
            raise ValueError("Variance must be positive")

    def generate_basic_dataset(self):
        x = np.random.normal(0, np.sqrt(self.variance), self.length)

        if self.changepoint is not None:
            x[self.changepoint + 1 :] += 2 * self.delta

        return x

    def generate_left_calibration_dataset(self):
        x_main = self.generate_basic_dataset()

        x_cal_pre = np.random.normal(0, np.sqrt(self.variance), self.calibration_size)

        return x_main, x_cal_pre

    def generate_right_calibration_dataset(self):
        x_main = self.generate_basic_dataset()

        x_cal_post = np.random.normal(
            2 * self.delta, np.sqrt(self.variance), self.calibration_size
        )

        return x_main, x_cal_post

    def generate_dual_calibration_dataset(self):
        x_main = self.generate_basic_dataset()

        x_cal_pre = np.random.normal(0, np.sqrt(self.variance), self.calibration_size)
        x_cal_post = np.random.normal(
            2 * self.delta, np.sqrt(self.variance), self.calibration_size
        )

        return x_main, x_cal_pre, x_cal_post

    def likelihood_ratio(self, z):
        return np.exp(
            (-((z - self.delta) ** 2) + (z + self.delta) ** 2) / (2 * self.variance)
        )

    def compute_scores(self):
        lr_main = self.likelihood_ratio(self.x)

        self.left_score = np.zeros((self.length, self.length))
        self.right_score = np.zeros((self.length, self.length))

        for t in range(self.length - 1):
            self.left_score[t, : t + 1] = lr_main[: t + 1]
            self.right_score[t, t + 1 :] = 1 / lr_main[t + 1 :]

        if self.calibration_mode in ["left", "both"]:
            lr_cal_pre = self.likelihood_ratio(self.x_cal_pre)
            self.left_score_cal = np.zeros((self.length, self.calibration_size))

            for t in range(self.length - 1):
                self.left_score_cal[t, :] = lr_cal_pre

        if self.calibration_mode in ["right", "both"]:
            lr_cal_post = self.likelihood_ratio(self.x_cal_post)
            self.right_score_cal = np.zeros((self.length, self.calibration_size))

            for t in range(self.length - 1):
                self.right_score_cal[t, :] = 1 / lr_cal_post

    def get_dataset(self):
        return self.x

    def get_left_score(self):
        return self.left_score

    def get_right_score(self):
        return self.right_score

    def get_calibration_data(self):
        calibration_data = {"mode": self.calibration_mode}

        if self.calibration_mode in ["left", "both"] and hasattr(
            self, "left_score_cal"
        ):
            calibration_data["left_cal_data"] = self.x_cal_pre
            calibration_data["left_cal_scores"] = self.left_score_cal

        if self.calibration_mode in ["right", "both"] and hasattr(
            self, "right_score_cal"
        ):
            calibration_data["right_cal_data"] = self.x_cal_post
            calibration_data["right_cal_scores"] = self.right_score_cal

        return calibration_data

    def visualize_samples(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(np.arange(self.length), self.x, "o-", alpha=0.6, ms=3, label="Data")

        if self.changepoint is not None:
            ax.axvline(
                self.changepoint,
                color="red",
                linestyle="--",
                label=f"Changepoint ($\\xi={self.changepoint}$)",
            )

        if self.changepoint is not None:
            ax.axhline(
                0,
                color="blue",
                linestyle="-",
                linewidth=1,
                alpha=0.5,
                label="Pre-change mean",
            )
            ax.axhline(
                2 * self.delta,
                color="green",
                linestyle="-",
                linewidth=1,
                alpha=0.5,
                label="Post-change mean",
            )

        ax.set_xlabel("Position $t$")
        ax.set_ylabel("Value")
        ax.set_title(f"Gaussian Mean Change ($\\delta={self.delta}$)")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig
