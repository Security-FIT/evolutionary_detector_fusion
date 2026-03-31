#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import det_curve


def calculate_EER(labels, scores) -> float:
    """
    Calculate the Equal Error Rate (EER) from the labels and predictions
    """
    fpr, fnr, _ = det_curve(labels, scores, pos_label=0)

    # eer from fpr and fnr can differ a bit (its an approximation), so we compute both and take the average
    eer_fpr = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_fnr = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_fpr + eer_fnr) / 2

    return eer


def get_num_params(model_name: str) -> int:
    """
    Return the number of parameters from a model name.
    """
    # Hardcoded, as we have only the model names
    param_map = {
        "HuBERT_base_MHFA_scores.txt": 96071338,
        "HuBERT_base_AASIST_scores.txt": 95279498,
        "HuBERT_base_SLS_scores.txt": 107915973,
        "HuBERT_large_MHFA_scores.txt": 320554194,
        "HuBERT_large_AASIST_scores.txt": 316704074,
        "HuBERT_large_SLS_scores.txt": 339489669,
        "HuBERT_extralarge_MHFA_scores.txt": 977048866,
        "HuBERT_extralarge_AASIST_scores.txt": 964205834,
        "HuBERT_extralarge_SLS_scores.txt": 1000056389,
        "Wav2Vec2_base_MHFA_scores.txt": 96071338,
        "Wav2Vec2_base_AASIST_scores.txt": 95279498,
        "Wav2Vec2_base_SLS_scores.txt": 107915973,
        "Wav2Vec2_large_MHFA_scores.txt": 320548050,
        "Wav2Vec2_large_AASIST_scores.txt": 316697930,
        "Wav2Vec2_large_SLS_scores.txt": 339483525,
        "Wav2Vec2_LV60k_MHFA_scores.txt": 320557778,
        "Wav2Vec2_LV60k_AASIST_scores.txt": 316707658,
        "Wav2Vec2_LV60k_SLS_scores.txt": 339493253,
        "WavLM_base_MHFA_scores.txt": 96081562,
        "WavLM_base_AASIST_scores.txt": 95289722,
        "WavLM_base_SLS_scores.txt": 107926197,
        "WavLM_baseplus_MHFA_scores.txt": 96081562,
        "WavLM_baseplus_AASIST_scores.txt": 95289722,
        "WavLM_baseplus_SLS_scores.txt": 107926197,
        "WavLM_large_MHFA_scores.txt": 320572178,
        "WavLM_large_AASIST_scores.txt": 316722058,
        "WavLM_large_SLS_scores.txt": 339507653,
        "XLSR_300M_MHFA_scores.txt": 320557778,
        "XLSR_300M_AASIST_scores.txt": 316707658,
        "XLSR_300M_SLS_scores.txt": 339493253,
        "XLSR_1B_MHFA_scores.txt": 977052450,
        "XLSR_1B_AASIST_scores.txt": 964209418,
        "XLSR_1B_SLS_scores.txt": 1000059973,
        "XLSR_2B_MHFA_scores.txt": 2191997730,
        "XLSR_2B_AASIST_scores.txt": 2162437738,
        "XLSR_2B_SLS_scores.txt": 2243900453,
    }

    return param_map[model_name]


def calculate_minDCF(labels, predictions, p_target=0.95, c_miss=1, c_fa=10) -> float:
    """
    Calculate the minimum Detection Cost Function (minDCF)
    """
    far, frr, thresholds = det_curve(labels, predictions, pos_label=0)

    c_det = c_miss * frr * p_target + c_fa * far * (1 - p_target)
    min_c_det = np.min(c_det)

    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    
    return min_dcf
