# Evolutionary Multi-Objective Fusion of Deepfake Speech Detectors

This repository contains the implementation of score fusion techniques for deepfake speech detection systems. It primarily focuses on using **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** to find optimal combinations of scores from various classifier models (HuBERT, Wav2Vec2, WavLM, XLSR with AASIST/MHFA/SLS backends) to minimize **Equal Error Rate (EER)** and model complexity.

It also includes baselines using **Logistic Regression** and **Manual Fusion** (simple averaging).

## Project Structure

*   **`nsga.py`**: Implementation of the NSGA-II algorithm for optimization. Handles the evolutionary loop, non-dominated sorting, and selection.
*   **`ea.py`**: Contains evolutionary operators such as mutation (binary and real) and crossover functions.
*   **`measure.py`**: Script to run comprehensive experiments (grid search on hyperparameters) across different population sizes, mutation rates, and crossover rates. Generates results in `results/`.
*   **`logistic_regression.py`**: Implements Logistic Regression fusion as a comparison baseline.
*   **`manual_fusion.py`**: Implements manual fusion baselines (simple averaging of specific model subsets).
*   **`metrics.py`**: Utility functions for calculating metrics like EER and counting parameters.
*   **`compare.py`**: Script to compare results from different fusion methods (Manual, Logistic Regression, NSGA-II) and generate plots.
*   **`load_scores.py`**: Handles loading score files from the `scores/` directory
*   **`analyze.py`**: Scripts for analyzing results and generating plots.

## Requirements

Implemented in Python 3.13, should work in older/newer versions. Install the required Python packages using:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
*   `numpy`
*   `pandas`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `joblib`
*   `pytables`

## Usage

### 1. NSGA-II Fusions

To run a single instance of the NSGA-II algorithm with default settings, execute:

```bash
python nsga.py
```

To run a comprehensive set of experiments with different hyperparameters (grid search over population sizes, mutation rates, etc.):

```bash
python measure.py
```

This will populate the `results/` folder with JSON files containing the performance metrics of the evolved populations.

### 2. Logistic Regression Baseline

To run the logistic regression fusion:

```bash
python logistic_regression.py
```

### 3. Manual Fusion Baseline

To run manual fusion strategies:

```bash
python manual_fusion.py
```

### 4. Analyzing Results

To analyze the generated results and create visualizations:

```bash
python analyze.py
```

### 5. Comparing Methods

To compare the performance of baselines (Manual, Logistic Regression) against NSGA-II fusions:

```bash
python compare.py
```

## Methodology

*   **Binary NSGA-II:** Evolution of a binary encoded chromosome where `1` selects a model and `0` ignores it. Optimizes for subset selection.
*   **Real-valued NSGA-II:** Evolution of weights for a weighted sum of scores. Optimizes for weight tuning.
*   **Objectives:** The multi-objective optimization aims to:
    1.  Minimize EER (Equal Error Rate).
    2.  Minimize the number of parameters / complexity.
    
    and produces Pareto front of non-dominated candidate solutions of detector fusions.
