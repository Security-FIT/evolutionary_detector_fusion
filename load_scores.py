#!/usr/bin/env python3

import os
import pandas as pd

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_scores(file_path: str) -> pd.DataFrame:
    """
    Load scores from a CSV file and return them as a DataFrame.

    Args:
        file_path (str): Path to the CSV file containing scores.

    Returns:
        pd.DataFrame: DataFrame containing the scores.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path, header=None, names=['file', 'score', 'label'])
    
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("The CSV file is empty.")
    
    # Convert scores to float
    df['score'] = df['score'].astype(float)
    
    return df


def load_all_scores() -> pd.DataFrame:
    """
    Load all scores from the 'scores' directory and return them as a single DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing all scores.
    """
    # Get the list of all CSV files in the 'scores' directory
    files = [f for f in os.listdir('scores')]
    
    names = None
    scores = {}
    labels = None

    # Load each file and append the DataFrame to the list
    for i, file in enumerate(files):
        logger.info(f"Loading scores from {file}")
        df = load_scores(os.path.join('scores', file))
        df.sort_values(by='file', inplace=True)
        df.reset_index(drop=True, inplace=True)
        scores[file] = df["score"].astype(float)

        if names is None and labels is None:
            names = df["file"]
            labels = df["label"]

    # Create a DataFrame from the scores dictionary
    all_scores_df = pd.DataFrame(scores)
    all_scores_df["file"] = names
    all_scores_df["label"] = labels
    # all_scores_df.set_index("file", inplace=True)
    
    return all_scores_df


def load_all_scores_cached(
    cache_path: str = "scores/scores_cache.h5",
    cache_key: str = "scores"
) -> pd.DataFrame:
    """
    Load scores from cache if available, otherwise compute and store them.
    """
    if os.path.exists(cache_path):
        # logger.info(f"Loading scores from cache: {cache_path}")
        
        df = pd.read_hdf(cache_path, key=cache_key)
        assert df is not None and type(df) == pd.DataFrame, "Cache is empty or not a DataFrame"
        
        df["label"] = df["label"].astype(int)
        
        # logger.info(f"Loaded scores from cache.")
        return df

    logger.info("Cache not found. Loading scores from source...")
    df = load_all_scores()
    df.to_hdf(cache_path, key=cache_key)
    logger.info(f"Scores cached at: {cache_path}")
    return df
