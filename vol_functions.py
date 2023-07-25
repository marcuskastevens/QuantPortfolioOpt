'''
Library for volatilty and variance estimation functions. 

Supported Estimation Algorithms:
1) Exponentially-Weighted Volatility Estimation
2) HAR Volatility Estimation

Future Estimation Algorithms:
1) GARCH
2) TGARCH
3) Etc.
'''
import pandas as pd
import numpy as np

# -------------------------------- Volatility Models --------------------------------

def ewma_vol(returns: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Compute the exponentially-weighted moving average estimate of volatility.
    
    Args:
        returns (pd.DataFrame): historical returns.
        lookback (int, optional): exponentially-weighted daily returns with a "lookback" day center-of-mass.
        
    Returns:
        pd.Series: EWMA volatility estimates.
    """
            
    # Calculate volatility estimates
    vols = returns.ewm(span=lookback).std().iloc[-1]    
        
    return vols

def ewma_vol_manual(returns: pd.DataFrame, alpha: float = .95, span: float = None) -> pd.Series:
    """
    Without libraries, compute the exponentially-weighted moving average estimate of volatility.
    
    Args:
        returns (pd.DataFrame): historical returns.
        alpha (float, optional): decay factor.
        span (float, optional): represents the decay factor in terms of the number of observations. 
                                Specifically, span is defined as the number of periods required for 
                                the EWMA to span the entire range of the data.
        
    Returns:
        pd.Series: EWMA volatility estimates.
    """
    
    # If decay is specified in terms of number of observations
    if span:
        alpha = 1 - 2 / (1 + span)

    # Square returns        
    squared_returns = np.square(returns)

    # Compute weights for squared returns (must reverse numerical range to capture reverse time weighting)
    weights = pd.Series(np.power(alpha, np.arange(len(returns)-1, -1, -1)) * (1 - alpha), 
                        index=squared_returns.index)          

    # Calculate EWMA variance
    # ewma_variance = np.sum(weights * squared_returns) 
    ewma_variance = pd.Series() 
    for col, ret in squared_returns.items():
        ewma_variance[col] = np.sum(weights*ret)

    # Compute volatility as the square root of EWMA variance
    vols = np.sqrt(ewma_variance)

    return vols