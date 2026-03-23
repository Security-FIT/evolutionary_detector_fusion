#!/usr/bin/env python3

import numpy as np

from load_scores import load_all_scores_cached
from metrics import calculate_EER, get_num_params


class ManualFusion:
    """
    Do a manual fusion of the scores.

    Variant light: Combine the best AASIST, MHFA and SLS classifiers (3 model fusion)
    Variant heavy: Combine the best SSL models (12 model fusion)
    Variant all: Combine all models (36 model fusion)
    This is a simple average of the scores.
    """
    def __init__(self):
        """
        Initialize the Manual Fusion model with training data.
        """
        self.scores_df = load_all_scores_cached()
        self.model_names = np.array([col for col in self.scores_df.columns if col not in ["file", "label"]])

    def fusion_light(self):
        """
        Fuse the best AASIST, MHFA and SLS classifier (3 model fusion).
        """
        models = ["XLSR_1B_AASIST_scores.txt", "WavLM_large_MHFA_scores.txt", "XLSR_2B_SLS_scores.txt"]
        scores = self.scores_df[models].values.mean(axis=1)
        labels = self.scores_df["label"].to_numpy()
        eer = calculate_EER(labels, scores)
        num_params = 0
        for model_name in models:
            num_params += get_num_params(model_name)

        print(f"Manual Fusion light: EER = {eer*100:.2f}, num_params = {num_params:e}")
        return eer, num_params

    def fusion_heavy(self):
        """
        Fuse the best SSL+pooling combinations (12 model fusion).
        """
        models = [
            "HuBERT_base_MHFA_scores.txt",
            "HuBERT_large_SLS_scores.txt",
            "HuBERT_extralarge_AASIST_scores.txt",
            "Wav2Vec2_base_MHFA_scores.txt",
            "Wav2Vec2_large_MHFA_scores.txt",
            "Wav2Vec2_LV60k_MHFA_scores.txt",
            "XLSR_300M_MHFA_scores.txt",
            "XLSR_1B_SLS_scores.txt",
            "XLSR_2B_SLS_scores.txt",
            "WavLM_base_MHFA_scores.txt",
            "WavLM_baseplus_MHFA_scores.txt",
            "WavLM_large_MHFA_scores.txt",
        ]
        scores = self.scores_df[models].values.mean(axis=1)
        labels = self.scores_df["label"].to_numpy()
        eer = calculate_EER(labels, scores)
        num_params = 0
        for model_name in models:
            num_params += get_num_params(model_name)

        print(f"Manual Fusion heavy: EER = {eer*100:.2f}, num_params = {num_params:e}")
        return eer, num_params

    def fusion_all(self):
        """
        Fuse all models (36 model fusion).
        """
        scores = self.scores_df[self.model_names].values.mean(axis=1)
        labels = self.scores_df["label"].to_numpy()
        eer = calculate_EER(labels, scores)
        num_params = 0
        for model_name in self.model_names:
            num_params += get_num_params(model_name)

        print(f"Manual Fusion all:   EER = {eer*100:.2f}, num_params = {num_params:e}")
        return eer, num_params


if __name__ == "__main__":
    fusion = ManualFusion()
    fusion.fusion_light()
    fusion.fusion_heavy()
    fusion.fusion_all()
