import numpy as np
from scipy.stats import chi2
from tqdm import tqdm


class CalibratedConfidenceInterval:

    def __init__(self, dataset, test_fn):
        self.dataset = dataset
        self.test_fn = test_fn

        self.calibration_data = dataset.get_calibration_data()
        self.calibration_mode = self.calibration_data["mode"]

    def compute_test(self):
        x = self.dataset.get_dataset()
        scores_left = self.dataset.get_left_score()
        scores_right = self.dataset.get_right_score()

        n = len(x)
        self.test_p_values = np.empty((n - 1, 2))
        self.conformal_p_matrix = np.empty((n - 1, n))

        if self.calibration_mode == "none":
            self._compute_test_no_calibration(x, scores_left, scores_right)
        elif self.calibration_mode == "left":
            self._compute_test_left_calibration(x, scores_left, scores_right)
        elif self.calibration_mode == "right":
            self._compute_test_right_calibration(x, scores_left, scores_right)
        elif self.calibration_mode == "both":
            self._compute_test_dual_calibration(x, scores_left, scores_right)

    def _compute_test_no_calibration(self, x, scores_left, scores_right):
        n = len(x)
        p = np.empty(n)

        rng = np.random.RandomState(42)

        for t in tqdm(range(n - 1), desc="Computing p-values"):
            for r in range(t + 1):
                num_greater = np.sum(scores_left[t, : r + 1] > scores_left[t, r])
                num_equal = np.sum(scores_left[t, : r + 1] == scores_left[t, r])

                p[r] = (num_greater + rng.uniform(0, 1) * num_equal) / (r + 1)

            for r in range(t + 1, n):
                num_greater = np.sum(scores_right[t, r:] > scores_right[t, r])
                num_equal = np.sum(scores_right[t, r:] == scores_right[t, r])

                p[r] = (num_greater + rng.uniform(0, 1) * num_equal) / (n - r)

            self.conformal_p_matrix[t] = p

            self.test_p_values[t] = (
                self.test_fn(p[: t + 1]),
                self.test_fn(p[t + 1 :]),
            )

    def _compute_test_left_calibration(self, x, scores_left, scores_right):
        n = len(x)
        p = np.empty(n)

        scores_left_cal = self.calibration_data.get("left_cal_scores")
        calibration_size_pre = scores_left_cal.shape[1]

        rng = np.random.RandomState(42)

        for t in tqdm(range(n - 1), desc="Computing p-values with left calibration"):
            for r in range(t + 1):
                score_r = scores_left[t, r]

                main_less = np.sum(scores_left[t, : r + 1] < score_r)
                main_equal = np.sum(scores_left[t, : r + 1] == score_r)

                cal_less = np.sum(scores_left_cal[t, :] < score_r)
                cal_equal = np.sum(scores_left_cal[t, :] == score_r)

                p[r] = (
                    main_less + cal_less + rng.uniform(0, 1) * (main_equal + cal_equal)
                ) / (r + 1 + calibration_size_pre)

            for r in range(t + 1, n):
                num_greater = np.sum(scores_right[t, r:] > scores_right[t, r])
                num_equal = np.sum(scores_right[t, r:] == scores_right[t, r])

                p[r] = (num_greater + rng.uniform(0, 1) * num_equal) / (n - r)

            self.conformal_p_matrix[t] = p

            self.test_p_values[t] = (
                self.test_fn(p[: t + 1]),
                self.test_fn(p[t + 1 :]),
            )

    def _compute_test_right_calibration(self, x, scores_left, scores_right):
        n = len(x)
        p = np.empty(n)

        scores_right_cal = self.calibration_data.get("right_cal_scores")
        calibration_size_post = scores_right_cal.shape[1]

        rng = np.random.RandomState(42)

        for t in tqdm(range(n - 1), desc="Computing p-values with right calibration"):
            for r in range(t + 1):
                num_greater = np.sum(scores_left[t, : r + 1] > scores_left[t, r])
                num_equal = np.sum(scores_left[t, : r + 1] == scores_left[t, r])

                p[r] = (num_greater + rng.uniform(0, 1) * num_equal) / (r + 1)

            for r in range(t + 1, n):
                score_r = scores_right[t, r]

                main_less = np.sum(scores_right[t, r:] < score_r)
                main_equal = np.sum(scores_right[t, r:] == score_r)

                cal_less = np.sum(scores_right_cal[t, :] < score_r)
                cal_equal = np.sum(scores_right_cal[t, :] == score_r)

                p[r] = (
                    main_less + cal_less + rng.uniform(0, 1) * (main_equal + cal_equal)
                ) / (n - r + calibration_size_post)

            self.conformal_p_matrix[t] = p

            self.test_p_values[t] = (
                self.test_fn(p[: t + 1]),
                self.test_fn(p[t + 1 :]),
            )

    def _compute_test_dual_calibration(self, x, scores_left, scores_right):
        n = len(x)
        p = np.empty(n)

        scores_left_cal = self.calibration_data.get("left_cal_scores")
        scores_right_cal = self.calibration_data.get("right_cal_scores")
        calibration_size_pre = scores_left_cal.shape[1]
        calibration_size_post = scores_right_cal.shape[1]

        rng = np.random.RandomState(42)

        for t in tqdm(range(n - 1), desc="Computing p-values with dual calibration"):
            for r in range(t + 1):
                score_r = scores_left[t, r]

                main_less = np.sum(scores_left[t, : r + 1] < score_r)
                main_equal = np.sum(scores_left[t, : r + 1] == score_r)

                cal_less = np.sum(scores_left_cal[t, :] < score_r)
                cal_equal = np.sum(scores_left_cal[t, :] == score_r)

                p[r] = (
                    main_less + cal_less + rng.uniform(0, 1) * (main_equal + cal_equal)
                ) / (r + 1 + calibration_size_pre)

            for r in range(t + 1, n):
                score_r = scores_right[t, r]

                main_less = np.sum(scores_right[t, r:] < score_r)
                main_equal = np.sum(scores_right[t, r:] == score_r)

                cal_less = np.sum(scores_right_cal[t, :] < score_r)
                cal_equal = np.sum(scores_right_cal[t, :] == score_r)

                p[r] = (
                    main_less + cal_less + rng.uniform(0, 1) * (main_equal + cal_equal)
                ) / (n - r + calibration_size_post)

            self.conformal_p_matrix[t] = p

            self.test_p_values[t] = (
                self.test_fn(p[: t + 1]),
                self.test_fn(p[t + 1 :]),
            )

    def compute_statistics(self):
        pass

    def compute_threshold(self, alpha):
        pass

    def compute_confidence_interval(self):
        self.ci = np.argwhere(self.statistics <= self.threshold).flatten()

    def get_ci_endpoints(self):
        if self.ci.size == 0:
            return None
        return int(self.ci[0]), int(self.ci[-1])


class CalibratedFisher(CalibratedConfidenceInterval):

    def __init__(self, dataset, test_fn):
        super().__init__(dataset, test_fn)
        self.cdf = lambda z: chi2.cdf(z, 4)

    def compute_statistics(self):
        epsilon = 1e-10
        self.statistics = -2 * np.sum(
            np.log(np.clip(self.test_p_values, epsilon, 1)), axis=1
        )

    def compute_threshold(self, alpha):
        self.threshold = chi2.ppf(1 - alpha, 4)


class CalibratedMinimum(CalibratedConfidenceInterval):

    def __init__(self, dataset, test_fn):
        super().__init__(dataset, test_fn)
        self.cdf = lambda z: z**2

    def compute_statistics(self):
        self.statistics = 1 - np.min(self.test_p_values, axis=1)

    def compute_threshold(self, alpha):
        self.threshold = np.sqrt(1 - alpha)


class CalibratedBonferroni(CalibratedConfidenceInterval):

    def __init__(self, dataset, test_fn):
        super().__init__(dataset, test_fn)
        self.cdf = lambda z: z

    def compute_statistics(self):
        min_corrected_p = np.minimum(np.min(2 * self.test_p_values, axis=1), 1)
        self.statistics = 1 - min_corrected_p

    def compute_threshold(self, alpha):
        self.threshold = 1 - alpha
