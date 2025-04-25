import numpy as np
import torch
import random
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset


class SentimentChangepointDataset:
    def __init__(
        self,
        length,
        changepoint,
        mixed_mode=False,
        pre_pos_ratio=1.0,
        post_pos_ratio=0.0,
        calibration_mode="none",
        calibration_size=100,
        dataset_name="sst2",
        device="cpu",
    ):
        self.length = length
        self.changepoint = changepoint
        self.mixed_mode = mixed_mode
        self.pre_pos_ratio = pre_pos_ratio
        self.post_pos_ratio = post_pos_ratio
        self.calibration_mode = calibration_mode
        self.calibration_size = calibration_size
        self.dataset_name = dataset_name
        self.device = device

        self.model, self.tokenizer = self.get_pretrained_sentiment_model(device)

        if calibration_mode == "none":
            self.texts, self.true_labels = self.generate_sentiment_dataset()
            self.x = np.array(self.true_labels)
            self.x_cal_pre = None
            self.x_cal_post = None
        elif calibration_mode == "left":
            (self.texts, self.true_labels), self.x_cal_pre = (
                self.generate_left_calibration_dataset()
            )
            self.x = np.array(self.true_labels)
            self.x_cal_post = None
        elif calibration_mode == "right":
            (self.texts, self.true_labels), self.x_cal_post = (
                self.generate_right_calibration_dataset()
            )
            self.x = np.array(self.true_labels)
            self.x_cal_pre = None
        elif calibration_mode == "both":
            (self.texts, self.true_labels), self.x_cal_pre, self.x_cal_post = (
                self.generate_dual_calibration_dataset()
            )
            self.x = np.array(self.true_labels)
        else:
            raise ValueError(f"Invalid calibration mode: {calibration_mode}")

        self.prepare_model_outputs()
        self.compute_scores()

    def get_pretrained_sentiment_model(self, device="cpu"):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        model.to(device)
        model.eval()
        return model, tokenizer

    def predict_sentiment(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu()
            predicted = probs.argmax(dim=1).item()

        return (predicted, probs.squeeze())

    def generate_sentiment_dataset(self):
        dataset = load_dataset(self.dataset_name)

        train_data = dataset["train"]
        positive_texts = [item["sentence"] for item in train_data if item["label"] == 1]
        negative_texts = [item["sentence"] for item in train_data if item["label"] == 0]

        random.shuffle(positive_texts)
        random.shuffle(negative_texts)

        n_pre = self.changepoint + 1
        n_post = self.length - n_pre

        if self.mixed_mode:
            n_pos_pre = int(n_pre * self.pre_pos_ratio)
            n_neg_pre = n_pre - n_pos_pre

            n_pos_post = int(n_post * self.post_pos_ratio)
            n_neg_post = n_post - n_pos_post

            if n_pos_pre + n_pos_post > len(
                positive_texts
            ) or n_neg_pre + n_neg_post > len(negative_texts):
                raise ValueError(
                    "Insufficient texts for the specified distribution and length."
                )

            pre_pos_texts = positive_texts[:n_pos_pre]
            pre_neg_texts = negative_texts[:n_neg_pre]
            pre_texts = pre_pos_texts + pre_neg_texts
            pre_labels = [1] * n_pos_pre + [0] * n_neg_pre

            pre_combined = list(zip(pre_texts, pre_labels))
            random.shuffle(pre_combined)
            pre_texts, pre_labels = zip(*pre_combined)

            post_pos_texts = positive_texts[n_pos_pre : n_pos_pre + n_pos_post]
            post_neg_texts = negative_texts[n_neg_pre : n_neg_pre + n_neg_post]
            post_texts = post_pos_texts + post_neg_texts
            post_labels = [1] * n_pos_post + [0] * n_neg_post

            post_combined = list(zip(post_texts, post_labels))
            random.shuffle(post_combined)
            post_texts, post_labels = zip(*post_combined)

            texts = list(pre_texts) + list(post_texts)
            true_labels = list(pre_labels) + list(post_labels)
        else:
            pre_label = 1 if self.pre_pos_ratio > 0.5 else 0
            post_label = 1 if self.post_pos_ratio > 0.5 else 0

            pre_texts = (
                positive_texts[:n_pre] if pre_label == 1 else negative_texts[:n_pre]
            )
            post_texts = (
                positive_texts[:n_post] if post_label == 1 else negative_texts[:n_post]
            )

            if (
                pre_label == 1
                and post_label == 1
                and n_pre + n_post > len(positive_texts)
            ):
                raise ValueError(
                    "Insufficient positive texts for the specified length."
                )
            if (
                pre_label == 0
                and post_label == 0
                and n_pre + n_post > len(negative_texts)
            ):
                raise ValueError(
                    "Insufficient negative texts for the specified length."
                )
            if pre_label == 1 and n_pre > len(positive_texts):
                raise ValueError("Insufficient positive texts for pre-change segment.")
            if post_label == 1 and n_post > len(positive_texts):
                raise ValueError("Insufficient positive texts for post-change segment.")
            if pre_label == 0 and n_pre > len(negative_texts):
                raise ValueError("Insufficient negative texts for pre-change segment.")
            if post_label == 0 and n_post > len(negative_texts):
                raise ValueError("Insufficient negative texts for post-change segment.")

            texts = pre_texts + post_texts
            true_labels = [pre_label] * n_pre + [post_label] * n_post

        return texts, true_labels

    def generate_left_calibration_dataset(self):
        dataset = load_dataset(self.dataset_name)

        train_data = dataset["train"]
        positive_texts = [item["sentence"] for item in train_data if item["label"] == 1]
        negative_texts = [item["sentence"] for item in train_data if item["label"] == 0]

        random.shuffle(positive_texts)
        random.shuffle(negative_texts)

        main_data = self.generate_sentiment_dataset()

        pre_labels = main_data[1][: self.changepoint + 1]
        dominant_pre = 1 if sum(pre_labels) / len(pre_labels) > 0.5 else 0

        if self.mixed_mode:
            n_pos_cal = int(self.calibration_size * self.pre_pos_ratio)
            n_neg_cal = self.calibration_size - n_pos_cal

            cal_pos_texts = positive_texts[-n_pos_cal:]
            cal_neg_texts = negative_texts[-n_neg_cal:]

            cal_texts = cal_pos_texts + cal_neg_texts
            cal_labels = [1] * n_pos_cal + [0] * n_neg_cal

            cal_combined = list(zip(cal_texts, cal_labels))
            random.shuffle(cal_combined)
            cal_texts, cal_labels = zip(*cal_combined)

            cal_data = (list(cal_texts), list(cal_labels))
        else:
            if dominant_pre == 1:
                cal_texts = positive_texts[-self.calibration_size :]
            else:
                cal_texts = negative_texts[-self.calibration_size :]

            cal_labels = [dominant_pre] * self.calibration_size
            cal_data = (cal_texts, cal_labels)

        return main_data, cal_data

    def generate_right_calibration_dataset(self):
        dataset = load_dataset(self.dataset_name)

        train_data = dataset["train"]
        positive_texts = [item["sentence"] for item in train_data if item["label"] == 1]
        negative_texts = [item["sentence"] for item in train_data if item["label"] == 0]

        random.shuffle(positive_texts)
        random.shuffle(negative_texts)

        main_data = self.generate_sentiment_dataset()

        post_labels = main_data[1][self.changepoint + 1 :]
        dominant_post = 1 if sum(post_labels) / len(post_labels) > 0.5 else 0

        if self.mixed_mode:
            n_pos_cal = int(self.calibration_size * self.post_pos_ratio)
            n_neg_cal = self.calibration_size - n_pos_cal

            cal_pos_texts = positive_texts[-n_pos_cal:]
            cal_neg_texts = negative_texts[-n_neg_cal:]

            cal_texts = cal_pos_texts + cal_neg_texts
            cal_labels = [1] * n_pos_cal + [0] * n_neg_cal

            cal_combined = list(zip(cal_texts, cal_labels))
            random.shuffle(cal_combined)
            cal_texts, cal_labels = zip(*cal_combined)

            cal_data = (list(cal_texts), list(cal_labels))
        else:
            if dominant_post == 1:
                cal_texts = positive_texts[-self.calibration_size :]
            else:
                cal_texts = negative_texts[-self.calibration_size :]

            cal_labels = [dominant_post] * self.calibration_size
            cal_data = (cal_texts, cal_labels)

        return main_data, cal_data

    def generate_dual_calibration_dataset(self):
        dataset = load_dataset(self.dataset_name)

        train_data = dataset["train"]
        positive_texts = [item["sentence"] for item in train_data if item["label"] == 1]
        negative_texts = [item["sentence"] for item in train_data if item["label"] == 0]

        random.shuffle(positive_texts)
        random.shuffle(negative_texts)

        main_data = self.generate_sentiment_dataset()

        pre_labels = main_data[1][: self.changepoint + 1]
        post_labels = main_data[1][self.changepoint + 1 :]
        dominant_pre = 1 if sum(pre_labels) / len(pre_labels) > 0.5 else 0
        dominant_post = 1 if sum(post_labels) / len(post_labels) > 0.5 else 0

        if self.mixed_mode:
            n_pos_cal_left = int(self.calibration_size * self.pre_pos_ratio)
            n_neg_cal_left = self.calibration_size - n_pos_cal_left

            cal_pos_texts_left = positive_texts[
                -n_pos_cal_left - self.calibration_size : -self.calibration_size
            ]
            cal_neg_texts_left = negative_texts[
                -n_neg_cal_left - self.calibration_size : -self.calibration_size
            ]

            cal_texts_left = cal_pos_texts_left + cal_neg_texts_left
            cal_labels_left = [1] * n_pos_cal_left + [0] * n_neg_cal_left

            cal_combined_left = list(zip(cal_texts_left, cal_labels_left))
            random.shuffle(cal_combined_left)
            cal_texts_left, cal_labels_left = zip(*cal_combined_left)

            cal_data_left = (list(cal_texts_left), list(cal_labels_left))

            n_pos_cal_right = int(self.calibration_size * self.post_pos_ratio)
            n_neg_cal_right = self.calibration_size - n_pos_cal_right

            cal_pos_texts_right = positive_texts[-n_pos_cal_right:]
            cal_neg_texts_right = negative_texts[-n_neg_cal_right:]

            cal_texts_right = cal_pos_texts_right + cal_neg_texts_right
            cal_labels_right = [1] * n_pos_cal_right + [0] * n_neg_cal_right

            cal_combined_right = list(zip(cal_texts_right, cal_labels_right))
            random.shuffle(cal_combined_right)
            cal_texts_right, cal_labels_right = zip(*cal_combined_right)

            cal_data_right = (list(cal_texts_right), list(cal_labels_right))
        else:
            if dominant_pre == 1:
                cal_texts_left = positive_texts[
                    -2 * self.calibration_size : -self.calibration_size
                ]
            else:
                cal_texts_left = negative_texts[
                    -2 * self.calibration_size : -self.calibration_size
                ]

            cal_labels_left = [dominant_pre] * self.calibration_size
            cal_data_left = (cal_texts_left, cal_labels_left)

            if dominant_post == 1:
                cal_texts_right = positive_texts[-self.calibration_size :]
            else:
                cal_texts_right = negative_texts[-self.calibration_size :]

            cal_labels_right = [dominant_post] * self.calibration_size
            cal_data_right = (cal_texts_right, cal_labels_right)

        return main_data, cal_data_left, cal_data_right

    def prepare_model_outputs(self):
        print("Preparing model outputs for main sequence...")
        predictions = []
        probabilities = []

        for i, text in enumerate(tqdm(self.texts)):
            pred, prob = self.predict_sentiment(text)
            predictions.append(pred)
            probabilities.append(prob)

        self.predictions = predictions
        self.probabilities = torch.stack(probabilities)

        if self.calibration_mode in ["left", "both"]:
            print("Preparing model outputs for left calibration data...")
            cal_pre_texts, cal_pre_labels = self.x_cal_pre

            predictions_cal_pre = []
            probabilities_cal_pre = []

            for i, text in enumerate(tqdm(cal_pre_texts)):
                pred, prob = self.predict_sentiment(text)
                predictions_cal_pre.append(pred)
                probabilities_cal_pre.append(prob)

            self.predictions_cal_pre = predictions_cal_pre
            self.probabilities_cal_pre = torch.stack(probabilities_cal_pre)

        if self.calibration_mode in ["right", "both"]:
            print("Preparing model outputs for right calibration data...")
            cal_post_texts, cal_post_labels = self.x_cal_post

            predictions_cal_post = []
            probabilities_cal_post = []

            for i, text in enumerate(tqdm(cal_post_texts)):
                pred, prob = self.predict_sentiment(text)
                predictions_cal_post.append(pred)
                probabilities_cal_post.append(prob)

            self.predictions_cal_post = predictions_cal_post
            self.probabilities_cal_post = torch.stack(probabilities_cal_post)

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
        seen_sentiments = {0: 0, 1: 0}

        for t in range(self.length):
            seen_sentiments[self.predictions[t]] += 1
            curr_sentiment = max(seen_sentiments, key=seen_sentiments.get)

            self.left_score[t, : t + 1] = self.probabilities[
                : t + 1, curr_sentiment
            ].cpu() / (1 - self.probabilities[: t + 1, curr_sentiment].cpu())

        self.right_score = np.zeros((self.length, self.length))
        seen_sentiments = {0: 0, 1: 0}

        for i in range(self.length - 1, -1, -1):
            t = self.length - i - 1
            seen_sentiments[self.predictions[i]] += 1
            curr_sentiment = max(seen_sentiments, key=seen_sentiments.get)

            self.right_score[t, t:] = self.probabilities[t:, curr_sentiment].cpu() / (
                1 - self.probabilities[t:, curr_sentiment].cpu()
            )

    def compute_scores_with_left_calibration(self):
        self.left_score = np.zeros((self.length, self.length))
        self.left_score_cal = np.zeros((self.length, self.calibration_size))

        seen_sentiments = {0: 0, 1: 0}

        for i, pred in enumerate(self.predictions_cal_pre):
            seen_sentiments[pred] += 1

        for t in range(self.length):
            seen_sentiments[self.predictions[t]] += 1
            curr_sentiment = max(seen_sentiments, key=seen_sentiments.get)

            self.left_score[t, : t + 1] = self.probabilities[
                : t + 1, curr_sentiment
            ].cpu() / (1 - self.probabilities[: t + 1, curr_sentiment].cpu())

            self.left_score_cal[t, :] = self.probabilities_cal_pre[
                :, curr_sentiment
            ].cpu() / (1 - self.probabilities_cal_pre[:, curr_sentiment].cpu())

        self.right_score = np.zeros((self.length, self.length))
        seen_sentiments = {0: 0, 1: 0}

        for i in range(self.length - 1, -1, -1):
            t = self.length - i - 1
            seen_sentiments[self.predictions[i]] += 1
            curr_sentiment = max(seen_sentiments, key=seen_sentiments.get)

            self.right_score[t, t:] = self.probabilities[t:, curr_sentiment].cpu() / (
                1 - self.probabilities[t:, curr_sentiment].cpu()
            )

    def compute_scores_with_right_calibration(self):
        self.left_score = np.zeros((self.length, self.length))
        seen_sentiments = {0: 0, 1: 0}

        for t in range(self.length):
            seen_sentiments[self.predictions[t]] += 1
            curr_sentiment = max(seen_sentiments, key=seen_sentiments.get)

            self.left_score[t, : t + 1] = self.probabilities[
                : t + 1, curr_sentiment
            ].cpu() / (1 - self.probabilities[: t + 1, curr_sentiment].cpu())

        self.right_score = np.zeros((self.length, self.length))
        self.right_score_cal = np.zeros((self.length, self.calibration_size))

        seen_sentiments = {0: 0, 1: 0}

        for i, pred in enumerate(self.predictions_cal_post):
            seen_sentiments[pred] += 1

        for i in range(self.length - 1, -1, -1):
            t = self.length - i - 1
            seen_sentiments[self.predictions[i]] += 1
            curr_sentiment = max(seen_sentiments, key=seen_sentiments.get)

            self.right_score[t, t:] = self.probabilities[t:, curr_sentiment].cpu() / (
                1 - self.probabilities[t:, curr_sentiment].cpu()
            )

            self.right_score_cal[t, :] = self.probabilities_cal_post[
                :, curr_sentiment
            ].cpu() / (1 - self.probabilities_cal_post[:, curr_sentiment].cpu())

    def compute_scores_with_dual_calibration(self):
        self.left_score = np.zeros((self.length, self.length))
        self.left_score_cal = np.zeros((self.length, self.calibration_size))

        seen_sentiments = {0: 0, 1: 0}

        for i, pred in enumerate(self.predictions_cal_pre):
            seen_sentiments[pred] += 1

        for t in range(self.length):
            seen_sentiments[self.predictions[t]] += 1
            curr_sentiment = max(seen_sentiments, key=seen_sentiments.get)

            self.left_score[t, : t + 1] = self.probabilities[
                : t + 1, curr_sentiment
            ].cpu() / (1 - self.probabilities[: t + 1, curr_sentiment].cpu())

            self.left_score_cal[t, :] = self.probabilities_cal_pre[
                :, curr_sentiment
            ].cpu() / (1 - self.probabilities_cal_pre[:, curr_sentiment].cpu())

        self.right_score = np.zeros((self.length, self.length))
        self.right_score_cal = np.zeros((self.length, self.calibration_size))

        seen_sentiments = {0: 0, 1: 0}

        for i, pred in enumerate(self.predictions_cal_post):
            seen_sentiments[pred] += 1

        for i in range(self.length - 1, -1, -1):
            t = self.length - i - 1
            seen_sentiments[self.predictions[i]] += 1
            curr_sentiment = max(seen_sentiments, key=seen_sentiments.get)

            self.right_score[t, t:] = self.probabilities[t:, curr_sentiment].cpu() / (
                1 - self.probabilities[t:, curr_sentiment].cpu()
            )

            self.right_score_cal[t, :] = self.probabilities_cal_post[
                :, curr_sentiment
            ].cpu() / (1 - self.probabilities_cal_post[:, curr_sentiment].cpu())

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

    def visualize_samples(self, num_samples=3, save_path=None):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))

        text_content = "SST-2 Sentiment Dataset Samples\n\n"

        text_content += "Examples before changepoint:\n"
        text_content += "-" * 60 + "\n"

        pre_pos_count = sum(
            1 for i in range(self.changepoint + 1) if self.true_labels[i] == 1
        )
        pre_neg_count = (self.changepoint + 1) - pre_pos_count

        text_content += (
            f"Distribution: {pre_pos_count/(self.changepoint+1)*100:.1f}% positive, "
        )
        text_content += f"{pre_neg_count/(self.changepoint+1)*100:.1f}% negative\n\n"

        for i in range(num_samples):
            idx = np.random.randint(0, self.changepoint)
            text_content += f'Text {i+1}: "{self.texts[idx]}"\n'
            text_content += f"True label: {'Positive' if self.true_labels[idx] == 1 else 'Negative'}, "
            text_content += f"Predicted: {'Positive' if self.predictions[idx] == 1 else 'Negative'}\n"
            text_content += (
                f"Confidence: {self.probabilities[idx][self.predictions[idx]]:.4f}\n\n"
            )

        text_content += "Examples after changepoint:\n"
        text_content += "-" * 60 + "\n"

        post_pos_count = sum(
            1
            for i in range(self.changepoint + 1, self.length)
            if self.true_labels[i] == 1
        )
        post_neg_count = (self.length - self.changepoint - 1) - post_pos_count

        text_content += f"Distribution: {post_pos_count/(self.length-self.changepoint-1)*100:.1f}% positive, "
        text_content += (
            f"{post_neg_count/(self.length-self.changepoint-1)*100:.1f}% negative\n\n"
        )

        for i in range(num_samples):
            idx = np.random.randint(self.changepoint + 1, self.length)
            text_content += f'Text {i+1}: "{self.texts[idx]}"\n'
            text_content += f"True label: {'Positive' if self.true_labels[idx] == 1 else 'Negative'}, "
            text_content += f"Predicted: {'Positive' if self.predictions[idx] == 1 else 'Negative'}\n"
            text_content += (
                f"Confidence: {self.probabilities[idx][self.predictions[idx]]:.4f}\n\n"
            )

        ax.text(
            0.01,
            0.99,
            text_content,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
            family="monospace",
        )

        ax.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        return fig
