'''
Utility functions for QuantPortfolioOpt -- used for general data transformations, cleaning, and preprocessing.

'''

import pandas as pd
import numpy as np

def drop_columns_with_nan(df: pd.DataFrame, n=10) -> pd.DataFrame:
    """
    Drops columns whose most recent n samples are NaN.

    Args:
        df (pd.DataFrame): input pd.DataFrame (e.g., returns).

    Returns:
        pd.DataFrame: cleaned pd.DataFrame.
    """

    # Get the last n rows of the DataFrame
    last_n_rows = df.tail(n)    
    
    # Get full NaN columns 
    last_n_rows_bool = last_n_rows.isna().all()
    cols_to_drop = last_n_rows_bool[last_n_rows_bool==True].index

    # Drop NaN columns 
    cleaned_df = df.drop(columns=cols_to_drop)

    return cleaned_df


def drop_columns_below_min_length(df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """
    Drop columns from a pd.DataFrame if they contain fewer non-null samples than the specified minimum.

    Parameters:
        df (pd.DataFrame): input pd.DataFrame (e.g., returns).
        min_samples (int): min number of non-null samples required for a column to be retained.

    Returns:
        pd.DataFrame: cleaned pd.DataFrame.

    """

    # Iterate over each column and its corresponding series in the DataFrame
    for col, series in df.items():

        # Trailing "min_samples" periods -- this ensures sufficient num of trailing datapoints are detected
        # Check if the number of non-null samples in the series is less than the minimum
        # if series.tail(min_samples).dropna().shape[0] < min_samples:
        if series.dropna().shape[0] < min_samples:

            # Drop the column from the DataFrame
            df = df.drop(columns=col)

    return df


