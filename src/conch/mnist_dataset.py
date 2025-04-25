import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from scipy.stats import ks_1samp, uniform


class MNISTChangepointDataset:
    def __init__(
        self,
        length,
        changepoint,
        digit1=3,
        digit2=7,
        calibration_mode="none",
        calibration_size=100,
        device="cpu",
    ):
        self.length = length
        self.changepoint = changepoint
        self.digit1 = digit1
        self.digit2 = digit2
        self.calibration_mode = calibration_mode
        self.calibration_size = calibration_size
        self.device = device

        if calibration_mode == "none":
            print(
                f"Generating MNIST dataset with {digit1} → {digit2} change at position {changepoint}"
            )
            self.x = self.generate_mnist_dataset(length, changepoint, digit1, digit2)
            self.x_cal_pre = None
            self.x_cal_post = None
        elif calibration_mode == "left":
            print(f"Generating MNIST dataset with left calibration ({digit1})")
            self.x, self.x_cal_pre = self.generate_left_calibration_dataset()
            self.x_cal_post = None
        elif calibration_mode == "right":
            print(f"Generating MNIST dataset with right calibration ({digit2})")
            self.x, self.x_cal_post = self.generate_right_calibration_dataset()
            self.x_cal_pre = None
        elif calibration_mode == "both":
            print(
                f"Generating MNIST dataset with dual calibration ({digit1} and {digit2})"
            )
            self.x, self.x_cal_pre, self.x_cal_post = (
                self.generate_dual_calibration_dataset()
            )
        else:
            raise ValueError(f"Invalid calibration mode: {calibration_mode}")

        print("Loading/training MNIST model...")
        self.model = self.get_mnist_trained_model(device)

        print("Preparing model outputs...")
        self.prepare_model_outputs()

        print("Computing scores...")
        self.compute_scores()

    def generate_mnist_dataset(self, length, changepoint, digit1=3, digit2=7):
        transform = transforms.ToTensor()
        mnist_data = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        data = mnist_data.data.numpy()
        targets = mnist_data.targets.numpy()

        images_digit1 = data[targets == digit1]
        images_digit2 = data[targets == digit2]
        np.random.shuffle(images_digit1)
        np.random.shuffle(images_digit2)

        n1 = changepoint + 1
        n2 = length - n1
        if n1 > len(images_digit1) or n2 > len(images_digit2):
            raise ValueError("Insufficient images for the specified digits and length.")

        data1 = images_digit1[:n1]
        data2 = images_digit2[:n2]
        x = np.concatenate([data1, data2], axis=0)

        x = x.reshape(length, -1).astype(np.float32) / 255.0
        return x

    def generate_left_calibration_dataset(self):
        transform = transforms.ToTensor()
        mnist_data = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        data = mnist_data.data.numpy()
        targets = mnist_data.targets.numpy()

        images_digit1 = data[targets == self.digit1]
        images_digit2 = data[targets == self.digit2]
        np.random.shuffle(images_digit1)
        np.random.shuffle(images_digit2)

        n1 = self.changepoint + 1
        n2 = self.length - n1

        n_cal = self.calibration_size

        if n1 > len(images_digit1) - n_cal or n2 > len(images_digit2):
            raise ValueError(
                "Insufficient images for the specified digits, length, and calibration."
            )

        main_pre = images_digit1[:n1]
        main_post = images_digit2[:n2]

        cal_pre = images_digit1[len(images_digit1) - n_cal :]

        x_main = np.concatenate([main_pre, main_post], axis=0)

        x_main = x_main.reshape(self.length, -1).astype(np.float32) / 255.0
        x_calibration_pre = cal_pre.reshape(n_cal, -1).astype(np.float32) / 255.0

        return x_main, x_calibration_pre

    def generate_right_calibration_dataset(self):
        transform = transforms.ToTensor()
        mnist_data = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        data = mnist_data.data.numpy()
        targets = mnist_data.targets.numpy()

        images_digit1 = data[targets == self.digit1]
        images_digit2 = data[targets == self.digit2]
        np.random.shuffle(images_digit1)
        np.random.shuffle(images_digit2)

        n1 = self.changepoint + 1
        n2 = self.length - n1

        n_cal = self.calibration_size

        if n1 > len(images_digit1) or n2 > len(images_digit2) - n_cal:
            raise ValueError(
                "Insufficient images for the specified digits, length, and calibration."
            )

        main_pre = images_digit1[:n1]
        main_post = images_digit2[:n2]

        cal_post = images_digit2[len(images_digit2) - n_cal :]

        x_main = np.concatenate([main_pre, main_post], axis=0)

        x_main = x_main.reshape(self.length, -1).astype(np.float32) / 255.0
        x_calibration_post = cal_post.reshape(n_cal, -1).astype(np.float32) / 255.0

        return x_main, x_calibration_post

    def generate_dual_calibration_dataset(self):
        transform = transforms.ToTensor()
        mnist_data = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        data = mnist_data.data.numpy()
        targets = mnist_data.targets.numpy()

        images_digit1 = data[targets == self.digit1]
        images_digit2 = data[targets == self.digit2]
        np.random.shuffle(images_digit1)
        np.random.shuffle(images_digit2)

        n1 = self.changepoint + 1
        n2 = self.length - n1

        n_cal = self.calibration_size

        if n1 > len(images_digit1) - n_cal or n2 > len(images_digit2) - n_cal:
            raise ValueError(
                "Insufficient images for the specified digits, length, and calibration."
            )

        main_pre = images_digit1[:n1]
        main_post = images_digit2[:n2]

        cal_pre = images_digit1[len(images_digit1) - n_cal :]
        cal_post = images_digit2[len(images_digit2) - n_cal :]

        x_main = np.concatenate([main_pre, main_post], axis=0)

        x_main = x_main.reshape(self.length, -1).astype(np.float32) / 255.0
        x_calibration_pre = cal_pre.reshape(n_cal, -1).astype(np.float32) / 255.0
        x_calibration_post = cal_post.reshape(n_cal, -1).astype(np.float32) / 255.0

        return x_main, x_calibration_pre, x_calibration_post

    def get_mnist_trained_model(self, device="cpu"):
        class MNISTModel(torch.nn.Module):
            def __init__(self):
                super(MNISTModel, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = torch.nn.Dropout(0.25)
                self.dropout2 = torch.nn.Dropout(0.5)
                self.fc1 = torch.nn.Linear(9216, 128)
                self.fc2 = torch.nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = torch.nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                return x

        model = MNISTModel().to(device)

        try:
            model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
            print("Loaded pre-trained MNIST model.")
            model.eval()
            return model
        except:
            print("Training a new MNIST model...")

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        print("Training MNIST model...")
        model.train()
        for epoch in range(1):
            for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} "
                        f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )

        test_dataset = MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        print(f"Test accuracy: {accuracy:.2f}%")

        torch.save(model.state_dict(), "mnist_model.pth")
        print("Model saved to mnist_model.pth")

        model.eval()
        return model

    def predict_digit(self, image):
        image_reshaped = image.reshape(1, 1, 28, 28)

        mean, std = 0.1307, 0.3081
        image_normalized = (image_reshaped - mean) / std

        image_tensor = torch.tensor(image_normalized, device=self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1).cpu()
            predicted = probs.argmax(dim=1).item()

        return (predicted, probs)

    def prepare_model_outputs(self):
        predicted_digits = []
        probabilities = []

        for i in tqdm(range(self.length), desc="Processing main dataset"):
            pred, prob = self.predict_digit(self.x[i])
            predicted_digits.append(pred)
            probabilities.append(prob)

        self.predicted_digits = predicted_digits
        self.probabilities = torch.vstack(probabilities)

        if self.calibration_mode in ["left", "both"]:
            predicted_cal_pre = []
            probabilities_cal_pre = []

            for i in tqdm(
                range(len(self.x_cal_pre)), desc="Processing left calibration"
            ):
                pred, prob = self.predict_digit(self.x_cal_pre[i])
                predicted_cal_pre.append(pred)
                probabilities_cal_pre.append(prob)

            self.predicted_cal_pre = predicted_cal_pre
            self.probabilities_cal_pre = torch.vstack(probabilities_cal_pre)

        if self.calibration_mode in ["right", "both"]:
            predicted_cal_post = []
            probabilities_cal_post = []

            for i in tqdm(
                range(len(self.x_cal_post)), desc="Processing right calibration"
            ):
                pred, prob = self.predict_digit(self.x_cal_post[i])
                predicted_cal_post.append(pred)
                probabilities_cal_post.append(prob)

            self.predicted_cal_post = predicted_cal_post
            self.probabilities_cal_post = torch.vstack(probabilities_cal_post)

    def compute_scores(self):
        if self.calibration_mode == "none":
            self.compute_scores_without_calibration()
        elif self.calibration_mode == "left":
            self.compute_scores_with_left_calibration()
        elif self.calibration_mode == "right":
            self.compute_scores_with_right_calibration()
        elif self.calibration_mode == "both":
            self.compute_scores_with_dual_calibration()

    def compute_scores_without_calibration(self):
        self.left_score = np.zeros((self.length, self.length))
        seen_digits = {}

        epsilon = 1e-10

        for t in tqdm(range(self.length), desc="Computing left scores"):
            pred = self.predicted_digits[t]
            seen_digits[pred] = seen_digits.get(pred, 0) + 1
            curr_digit = max(seen_digits, key=seen_digits.get)
            probs_curr = self.probabilities[: t + 1, curr_digit].numpy()
            self.left_score[t, : t + 1] = probs_curr / (1 - probs_curr + epsilon)

        self.right_score = np.zeros((self.length, self.length))
        seen_digits = {}

        for i in tqdm(range(self.length), desc="Computing right scores"):
            t = self.length - i - 1
            pred = self.predicted_digits[t]
            seen_digits[pred] = seen_digits.get(pred, 0) + 1
            curr_digit = max(seen_digits, key=seen_digits.get)
            probs_curr = self.probabilities[t:, curr_digit].numpy()
            self.right_score[t, t:] = probs_curr / (1 - probs_curr + epsilon)

    def compute_scores_with_right_calibration(self):
        self.left_score = np.zeros((self.length, self.length))
        seen_digits = {}

        epsilon = 1e-10

        for t in tqdm(range(self.length), desc="Computing left scores"):
            pred = self.predicted_digits[t]
            seen_digits[pred] = seen_digits.get(pred, 0) + 1
            curr_digit = max(seen_digits, key=seen_digits.get)

            probs_curr = self.probabilities[: t + 1, curr_digit].numpy()
            self.left_score[t, : t + 1] = probs_curr / (1 - probs_curr + epsilon)

        self.right_score = np.zeros((self.length, self.length))
        self.right_score_cal = np.zeros((self.length, self.calibration_size))

        seen_digits = {}

        for i, pred in enumerate(self.predicted_cal_post):
            seen_digits[pred] = seen_digits.get(pred, 0) + 1

        for i in tqdm(
            range(self.length), desc="Computing right scores with calibration"
        ):
            t = self.length - i - 1
            pred = self.predicted_digits[t]
            seen_digits[pred] = seen_digits.get(pred, 0) + 1
            curr_digit = max(seen_digits, key=seen_digits.get)

            probs_curr_main = self.probabilities[t:, curr_digit].numpy()
            self.right_score[t, t:] = probs_curr_main / (1 - probs_curr_main + epsilon)

            probs_curr_cal = self.probabilities_cal_post[:, curr_digit].numpy()
            self.right_score_cal[t, :] = probs_curr_cal / (1 - probs_curr_cal + epsilon)

    def compute_scores_with_dual_calibration(self):
        self.left_score = np.zeros((self.length, self.length))
        self.left_score_cal = np.zeros((self.length, self.calibration_size))

        epsilon = 1e-10

        seen_digits = {}

        for i, pred in enumerate(self.predicted_cal_pre):
            seen_digits[pred] = seen_digits.get(pred, 0) + 1

        for t in tqdm(
            range(self.length), desc="Computing left scores with dual calibration"
        ):
            pred = self.predicted_digits[t]
            seen_digits[pred] = seen_digits.get(pred, 0) + 1
            curr_digit = max(seen_digits, key=seen_digits.get)

            probs_curr_main = self.probabilities[: t + 1, curr_digit].numpy()
            self.left_score[t, : t + 1] = probs_curr_main / (
                1 - probs_curr_main + epsilon
            )

            probs_curr_cal = self.probabilities_cal_pre[:, curr_digit].numpy()
            self.left_score_cal[t, :] = probs_curr_cal / (1 - probs_curr_cal + epsilon)

        self.right_score = np.zeros((self.length, self.length))
        self.right_score_cal = np.zeros((self.length, self.calibration_size))

        seen_digits = {}

        for i, pred in enumerate(self.predicted_cal_post):
            seen_digits[pred] = seen_digits.get(pred, 0) + 1

        for i in tqdm(
            range(self.length), desc="Computing right scores with dual calibration"
        ):
            t = self.length - i - 1
            pred = self.predicted_digits[t]
            seen_digits[pred] = seen_digits.get(pred, 0) + 1
            curr_digit = max(seen_digits, key=seen_digits.get)

            probs_curr_main = self.probabilities[t:, curr_digit].numpy()
            self.right_score[t, t:] = probs_curr_main / (1 - probs_curr_main + epsilon)

            probs_curr_cal = self.probabilities_cal_post[:, curr_digit].numpy()
            self.right_score_cal[t, :] = probs_curr_cal / (1 - probs_curr_cal + epsilon)

    def get_dataset(self):
        return self.x

    def get_left_score(self):
        return self.left_score

    def get_right_score(self):
        return self.right_score

    def get_calibration_data(self):
        calibration_data = {"mode": self.calibration_mode}

        if self.calibration_mode in ["left", "both"]:
            calibration_data["left_cal_data"] = self.x_cal_pre
            calibration_data["left_cal_scores"] = self.left_score_cal

        if self.calibration_mode in ["right", "both"]:
            calibration_data["right_cal_data"] = self.x_cal_post
            calibration_data["right_cal_scores"] = self.right_score_cal

        return calibration_data

    def get_discrepancy_scores(self):
        discrepancy_scores = np.empty(self.length - 1)
        statistics = []

        np.random.seed(42)

        for t in tqdm(range(self.length - 1), desc="Computing discrepancy scores"):
            p = np.empty(self.length)

            for r in range(t + 1):
                p[r] = (
                    np.count_nonzero(
                        self.left_score[t, : r + 1] > self.left_score[t, r]
                    )
                    + np.random.uniform(0, 1)
                    * np.count_nonzero(
                        self.left_score[t, : r + 1] == self.left_score[t, r]
                    )
                ) / (r + 1)

            for r in range(self.length - 1, t, -1):
                p[r] = (
                    np.count_nonzero(self.right_score[t, r:] > self.right_score[t, r])
                    + np.random.uniform(0, 1)
                    * np.count_nonzero(
                        self.right_score[t, r:] == self.right_score[t, r]
                    )
                ) / (self.length - r)

            left_ks = ks_1samp(p[: t + 1], uniform.cdf)
            right_ks = ks_1samp(p[t + 1 :], uniform.cdf)
            statistics.append((left_ks, right_ks))

            discrepancy_scores[t] = left_ks.statistic * np.sqrt(
                t + 1
            ) + right_ks.statistic * np.sqrt(self.length - t - 1)

        return discrepancy_scores, statistics

    def visualize_samples(self, num_samples=5, save_path=None):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 5))

        for i in range(num_samples):
            idx = min(i, self.changepoint)
            img = self.x[idx].reshape(28, 28)
            pred = self.predicted_digits[idx]

            axes[0, i].imshow(img, cmap="gray")
            axes[0, i].set_title(f"t={idx}, pred={pred}")
            axes[0, i].axis("off")

        for i in range(num_samples):
            idx = min(i, self.length - self.changepoint - 1) + self.changepoint + 1
            img = self.x[idx].reshape(28, 28)
            pred = self.predicted_digits[idx]

            axes[1, i].imshow(img, cmap="gray")
            axes[1, i].set_title(f"t={idx}, pred={pred}")
            axes[1, i].axis("off")

        plt.suptitle(
            f"MNIST Samples: Digit {self.digit1} → Digit {self.digit2} at t = {self.changepoint}"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig

    def visualize_changepoint_detection(self, save_path=None):
        import matplotlib.pyplot as plt

        discrepancy_scores, statistics = self.get_discrepancy_scores()

        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        ax1 = axes[0]
        predictions = np.array(self.predicted_digits)
        unique_digits = np.unique(predictions)
        for digit in unique_digits:
            indices = np.where(predictions == digit)[0]
            ax1.scatter(indices, np.ones_like(indices) * digit, label=f"Digit {digit}")

        ax1.axvline(
            self.changepoint,
            color="red",
            linestyle="--",
            label=f"True changepoint (t={self.changepoint})",
        )
        ax1.set_xlabel("Position (t)")
        ax1.set_ylabel("Predicted Digit")
        ax1.set_title("MNIST Digit Predictions")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = axes[1]
        ax2.plot(np.arange(1, len(discrepancy_scores) + 1), discrepancy_scores)
        ax2.axvline(
            self.changepoint,
            color="red",
            linestyle="--",
            label=f"True changepoint (t={self.changepoint})",
        )

        max_idx = np.argmax(discrepancy_scores)
        ax2.scatter(
            max_idx + 1,
            discrepancy_scores[max_idx],
            color="green",
            s=100,
            marker="o",
            label=f"Max score at t={max_idx+1}",
        )

        ax2.set_xlabel("Position (t)")
        ax2.set_ylabel("Discrepancy Score")
        ax2.set_title("Changepoint Detection Discrepancy Scores")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = axes[2]
        p_values = np.array(
            [min(stat[0].pvalue, stat[1].pvalue) for stat in statistics]
        )
        ax3.plot(np.arange(1, len(p_values) + 1), p_values, label="p-values")
        ax3.axvline(
            self.changepoint,
            color="red",
            linestyle="--",
            label=f"True changepoint (t={self.changepoint})",
        )
        ax3.axhline(0.05, color="green", linestyle=":", label="α=0.05")

        ax3.set_xlabel("Position (t)")
        ax3.set_ylabel("p-value")
        ax3.set_title("Uniformity Test p-values")
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig
