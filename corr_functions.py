'''
Library for correlation matrix estimation functions. 

Supported Estimation Algorithms:
1) Exponentially-Weighted Correlation Estimation

Future Estimation Algorithms:
1) DCC
2) De-Toning
'''

import pandas as pd
import numpy as np

# -------------------------------- Correlation Models --------------------------------
def ewma_corr(returns: pd.DataFrame, lookback: int = 150) -> pd.DataFrame:
    """
    Compute the exponentially-weighted correlation matrix estimation.
    
    Args:
        returns (pd.DataFrame): historical returns.
        lookback (int, optional): exponentially-weighted 3-day overlapping returns with a "lookback" day center-of-mass.
    
    Returns:
        pd.DataFrame: ewma correlation matrix.
    """

    n = returns.shape[1]

    # Calculate correlation estimates    
    corr = returns.ewm(span=lookback).corr().droplevel(0).iloc[-n:]
        
    return corr