#!/usr/bin/env python3

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Tuple

from load_scores import load_all_scores_cached
from metrics import calculate_EER, get_num_params

logger = logging.getLogger(__name__)

"""
Logistic regression fusion for binary classification to compare with the NSGA-II fusion.
"""

eer_map = {
    "HuBERT_base_AASIST_scores.txt": 13.51,
    "HuBERT_base_MHFA_scores.txt": 10.82,
    "HuBERT_base_SLS_scores.txt": 12.52,
    "HuBERT_large_AASIST_scores.txt": 8.06,
    "HuBERT_large_MHFA_scores.txt": 7.74,
    "HuBERT_large_SLS_scores.txt": 6.39,
    "HuBERT_extralarge_AASIST_scores.txt": 8.11,
    "HuBERT_extralarge_MHFA_scores.txt": 8.45,
    "HuBERT_extralarge_SLS_scores.txt": 8.88,
    "Wav2Vec2_base_AASIST_scores.txt": 16.65,
    "Wav2Vec2_base_MHFA_scores.txt": 12.42,
    "Wav2Vec2_base_SLS_scores.txt": 15.86,
    "Wav2Vec2_large_AASIST_scores.txt": 19.32,
    "Wav2Vec2_large_MHFA_scores.txt": 7.04,
    "Wav2Vec2_large_SLS_scores.txt": 7.78,
    "Wav2Vec2_LV60k_AASIST_scores.txt": 8.64,
    "Wav2Vec2_LV60k_MHFA_scores.txt": 8.22,
    "Wav2Vec2_LV60k_SLS_scores.txt": 9.09,
    "XLSR_300M_AASIST_scores.txt": 7.53,
    "XLSR_300M_MHFA_scores.txt": 5.34,
    "XLSR_300M_SLS_scores.txt": 6.68,
    "XLSR_1B_AASIST_scores.txt": 5.56,
    "XLSR_1B_MHFA_scores.txt": 6.13,
    "XLSR_1B_SLS_scores.txt": 5.09,
    "XLSR_2B_AASIST_scores.txt": 7.41,
    "XLSR_2B_MHFA_scores.txt": 6.28,
    "XLSR_2B_SLS_scores.txt": 4.79,
    "WavLM_base_AASIST_scores.txt": 12.30,
    "WavLM_base_MHFA_scores.txt": 8.35,
    "WavLM_base_SLS_scores.txt": 10.87,
    "WavLM_baseplus_AASIST_scores.txt": 10.37,
    "WavLM_baseplus_MHFA_scores.txt": 8.65,
    "WavLM_baseplus_SLS_scores.txt": 9.26,
    "WavLM_large_AASIST_scores.txt": 7.03,
    "WavLM_large_MHFA_scores.txt": 5.12,
    "WavLM_large_SLS_scores.txt": 5.77,
}


class LogisticRegressionFusion:
    def __init__(self):
        """
        Initialize the Logistic Regression model with training data.
        :param X_train: Training features
        :param y_train: Training labels
        """
        self.model = LogisticRegression()
        self.scores_df = load_all_scores_cached()
        self.scores_df["label"] = 1 - self.scores_df["label"]
        self.model_names = np.array([col for col in self.scores_df.columns if col not in ["file", "label"]])
        logger.info("Initialized LogisticRegressionFusion")

    def train(self, log=True):
        """
        Train the Logistic Regression model using the loaded scores.
        :return: None
        """
        if log == True:
            logger.info("Training Logistic Regression model")
        X_train = self.scores_df[self.model_names].values
        y_train = self.scores_df["label"].to_numpy()
        self.model.fit(X_train, y_train)
        if log == True:
            logger.info("Logistic Regression model trained")

    def predict_proba(self, X_test) -> np.ndarray:
        """
        Predict the labels for the test data.
        :param X_test: Test features
        :return: Predicted scores
        """
        return self.model.predict_proba(X_test)[:, 0]

    def evaluate_model(self, log=True) -> Tuple[float, int]:
        """
        Evaluate the Logistic Regression model using the loaded scores.
        :return: EER and number of parameters of used models
        """
        if log == True:
            logger.info("Evaluating Logistic Regression model")
        X_test = self.scores_df[self.model_names].values
        y_test = self.scores_df["label"].to_numpy()
        scores = self.predict_proba(X_test)
        eer = calculate_EER(y_test, scores)

        num_params = 0
        for model_name in self.model_names:
            num_params += get_num_params(model_name)

        return eer, num_params

    def _print_weights(self, sorted=True):
        """
        Print the weights of the Logistic Regression model.
        """
        if sorted:
            sorted_indices = np.argsort(np.abs(self.model.coef_.squeeze()))[::-1]
        else:
            sorted_indices = np.arange(len(self.model_names))

        for i in sorted_indices:
            max_length_model_name = len("HuBERT extralarge AASIST")  # Longest model name
            model_name = " ".join(self.model_names[i].split("_")[0:3])  # split for SSL and pooling
            weight = self.model.coef_.squeeze()[i]  # Get the weight of the model
            sign_space = (
                " " if weight >= 0 else ""
            )  # Add space to positive weights to align with negative weights
            print(
                f"{model_name}{' ' * (max_length_model_name - len(model_name))}: {sign_space}{weight:.4f} "
            )

    def remove_least_contributing_models(self, n=1):
        """
        Remove the least contributing model based on the weights of the Logistic Regression model.
        :param n: Number of least contributing models to remove
        :return: None
        """
        sorted_indices = np.argsort(np.abs(self.model.coef_.squeeze()))[::-1]
        least_contributing_models = self.model_names[sorted_indices[-n:]]

        # Remove model names
        self.model_names = np.delete(self.model_names, sorted_indices[-n:])
        # Remove columns from scores_df
        self.scores_df = self.scores_df.drop(columns=least_contributing_models)
        # Refit the model with the reduced set of features
        X_train = self.scores_df[self.model_names].values
        y_train = self.scores_df["label"].to_numpy()
        self.model.fit(X_train, y_train)

        # logger.info(f"Removed least contributing models: {least_contributing_models}")

    def remove_highest_eer_models(self, n=1):
        """
        Remove the models with the highest standalone EER from the LR fusion.
        :param n: Number of models with highest EER to remove
        :return: None
        """
        eer_values = [eer_map[model_name] for model_name in self.model_names]
        sorted_indices = np.argsort(eer_values)[::-1]
        highest_eer_models = self.model_names[sorted_indices[:n]]

        # Remove model names
        self.model_names = np.delete(self.model_names, sorted_indices[:n])
        # Remove columns from scores_df
        self.scores_df = self.scores_df.drop(columns=highest_eer_models)
        # Refit the model with the reduced set of features
        X_train = self.scores_df[self.model_names].values
        y_train = self.scores_df["label"].to_numpy()
        self.model.fit(X_train, y_train)

        logger.info(f"Removed highest EER models: {highest_eer_models}")



def evaluate_for_EER_vs_params():
    """
    Evaluate the Logistic Regression model for EER. Iteratively remove the least contributing models and evaluate the model.
    """
    fusions = []

    fusion = LogisticRegressionFusion()
    fusion.train()
    eers, num_models = [], []
    eer, num_params = fusion.evaluate_model()
    eers.append(eer)
    num_models.append(36)

    fusions.append(
        {
            "model_names": fusion.model_names.tolist(),
            "weights": fusion.model.coef_.squeeze().tolist(),
            "eer": eer,
            "num_params": num_params,
        }
    )
    print(f"EER or 36 models: {eer*100:.2f}%, Number of parameters: {num_params:e}")

    for i in range(1, 35):
        fusion.train(log=False)
        fusion.remove_least_contributing_models(n=1)
        # fusion.remove_highest_eer_models(n=1)
        eer, num_params = fusion.evaluate_model(log=False)
        print(f"EER of {36-i} models: {eer*100:.2f}%, Number of parameters: {num_params:e}")
        eers.append(eer)
        num_models.append(36 - i)
        fusions.append(
            {
                "model_names": fusion.model_names.tolist(),
                "weights": fusion.model.coef_.squeeze().tolist(),
                "eer": eer,
                "num_params": num_params,
            }
        )

    json.dump(fusions, open("logistic_regression_fusion.json", "w"), indent=2)

    # Plot EER vs Number of Models
    plt.figure(figsize=(10, 5))
    plt.plot(num_models, [eer * 100 for eer in eers], marker="o")
    plt.title("EER vs Number of Models")
    plt.xlabel("Number of Models")
    plt.ylabel("EER (%)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    evaluate_for_EER_vs_params()
    # fusion = LogisticRegressionFusion()
    # fusion.train()
    # print(fusion.model.coef_.tolist())
