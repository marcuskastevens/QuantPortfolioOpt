'''
Library for covariance estimation functions. Contains general wrapper risk 
matrix function, allowing you to run any risk model from one function.

Supported Estimation Algorithms:
1) Empirical/Historical Covariance
2) Shrunken Empirical Covariance
3) Trailing Covariance
4) Shrunken Trailing Covariance
5) Exponentially Weighted Covariance
6) Shrunken Exponentially Weighted Covariance


Future Estimation Algorithms:
1) Ledoit-Wolf Shrunken Covariance
2) De-Toning / Constant Residual Eigenvalue Method (MLDP)
3) RMT / RIE / Marcenko-Pastur EPO Risk Model
4) DDC Multivariate GARCH Covariance

'''

import pandas as pd
import numpy as np
from quant_tools.beta import utils, vol_functions, corr_functions

# -------------------------------- Constants --------------------------------
ROLLING_WINDOW = 3

# -------------------------------- Risk Models --------------------------------

def risk_matrix(returns: pd.DataFrame, method: str ="empirical_cov", min_samples: int = 21, apply_rolling_returns: bool = False, strict_min_samples: bool = False, **kwargs):
    """ 
    Wrapper function to compute the covariance matrix using the "risk model" supplied in the "method" parameter.

    Args:
        returns (pd.DataFrame): historical returns.
        method (str, optional): covariance risk model. Defaults to "empirical_cov".
        min_samples (int): minimum number of observations per column to form covariance matrix.
        apply_rolling_returns (bool): if true, applies rolling mean to mitigate effects of asynchronous trading in global asset classes.
        strict_min_samples (bool): if true, drop columns with fewer than ("lookback_cov" | "lookback_corr" | "lookback_vol") samples.
    """

    # Preprocess, clean, and inspect returns
    returns, kwargs = preprocess_risk_model_returns(returns=returns, 
                                                    method=method, 
                                                    min_samples=min_samples, 
                                                    apply_rolling_returns=apply_rolling_returns, 
                                                    strict_min_samples=strict_min_samples,
                                                    **kwargs)

    # Return risk (cov) matrix
    return risk_models[method](returns=returns, **kwargs)

def empirical_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """ 
    Computes full-sample empirical covariance matrix.

    Args:
        returns (pd.DataFrame): historical returns.

    Returns:
        pd.DataFrame: empirical covariance matrix.
    """

    # Compute empirical covariance matrix
    cov = returns.cov()

    if cov.isnull().values.any():
        print(cov)
        raise ValueError("Covariance matrix contains missing values")
    
    return cov

def shrunken_empirical_cov(returns: pd.DataFrame, returns_non_3d: pd.DataFrame, shrinkage: float = 0.1) -> pd.DataFrame:
    """ 
    Computes shrunken empirical covariance matrix.

    Args:
        returns (pd.DataFrame): historical returns.
        shrinkage (float, optional): percent shrinkage of off-diagonal correlations.
        min_samples (int): minimum number of observations per column to form covariance matrix.

    Returns:
        pd.DataFrame: shrunken empirical covariance matrix.
    """

    # Calculate volatility estimates
    vols = returns_non_3d.std()

    # Calculate correlation estimates
    corr = returns.corr()

    # Shrink off-diagonal correlations
    shrunken_corr = corr * (1 - shrinkage)
    np.fill_diagonal(shrunken_corr.values, 1)

    # Compute covariance matrix
    cov = corr_to_cov(corr=shrunken_corr, vols=vols)

    if cov.isnull().values.any():
        print(cov)
        raise ValueError("Covariance matrix contains missing values")
    
    return cov

def empirical_trailing_cov(returns: pd.DataFrame, lookback_cov: int = 150) -> pd.DataFrame:
    """ 
    Computes "lookback_cov" day empirical covariance matrix.

    Args:
        returns (pd.DataFrame): historical returns.
        lookback_cov (int, optional): covariance matrix with "lookback_cov" day center-of-mass.

    Returns:
        pd.DataFrame: trailing "lookback_cov" day empirical covariance matrix.
    """

    # Drop securities with insufficient data for estimatation 
    returns = utils.drop_columns_below_min_length(df=returns, min_samples=lookback_cov)

    # Compute trailing covariance matrix
    cov = returns.tail(lookback_cov).cov()

    if cov.isnull().values.any():
        print(cov)
        raise ValueError("Covariance matrix contains missing values")

    return cov

def shrunken_empirical_trailing_cov(returns: pd.DataFrame, returns_non_3d: pd.DataFrame, shrinkage: float = 0.1, lookback_vol: int = 60, lookback_corr: int = 150) -> pd.DataFrame:
    """
    Computes shrunken "lookback_cov" day empirical covariance matrix.

    Args:
        returns (pd.DataFrame): historical returns.
        shrinkage (float, optional): percent shrinkage of off-diagonal correlations.
        lookback_cov (int, optional): covariance matrix with "lookback_cov" day center-of-mass.

    Returns:
        pd.DataFrame: shrunken "lookback_cov" day empirical covariance matrix.
    """

    # Calculate volatility estimates
    vols = returns_non_3d.tail(lookback_vol).std()

    # Calculate correlation estimates
    corr = returns.tail(lookback_corr).corr()

    # Shrink off-diagonal correlations
    shrunken_corr = corr * (1 - shrinkage)
    np.fill_diagonal(shrunken_corr.values, 1)

    # Compute covariance matrix
    cov = corr_to_cov(corr=shrunken_corr, vols=vols)

    if cov.isnull().values.any():
        print(cov)
        raise ValueError("Covariance matrix contains missing values")
    
    return cov


def ewma_cov(returns: pd.DataFrame, returns_non_3d: pd.DataFrame, lookback_vol: int = 60, lookback_corr: int = 150) -> pd.DataFrame:
    """
    Compute the covariance matrix of returns based on the given methodology.
    
    Args:
        returns (pd.DataFrame): historical returns.
        lookback_vol (int, optional): exponentially-weighted daily returns with a "lookback_vol" day center-of-mass.
        lookback_corr (int, optional): exponentially-weighted 3-day overlapping returns with a "lookback_corr" day center-of-mass.
    
    Returns:
        pd.DataFrame: ewma covariance matrix.
    """

    n = returns.shape[1]

    # Calculate correlation estimates    
    corr = corr_functions.ewma_corr(returns=returns, lookback=lookback_corr)# returns.ewm(span=lookback_corr).corr().droplevel(0).iloc[-n:]
        
    # Calculate volatility estimates
    vols = vol_functions.ewma_vol(returns=returns_non_3d, lookback=lookback_vol) # returns_non_3d.ewm(span=lookback_vol).std().iloc[-1]
    
    # Compute covariance matrix
    cov = corr_to_cov(corr=corr, vols=vols)

    if cov.isnull().values.any():
        print(cov)
        raise ValueError("Covariance matrix contains missing values")
    
    return cov


def shrunken_ewma_cov(returns: pd.DataFrame, returns_non_3d: pd.DataFrame, shrinkage: float = 0.1, lookback_vol: int = 60, lookback_corr: int = 150) -> pd.DataFrame:
    """
    Compute the covariance matrix of returns based on the given methodology.
    
    Args:
        returns (pd.DataFrame): historical returns.
        shrinkage (float, optional): percent shrinkage of off-diagonal correlations.
        lookback_vol (int, optional): exponentially-weighted daily returns with a "lookback_vol" day center-of-mass.
        lookback_corr (int, optional): exponentially-weighted 3-day overlapping returns with a "lookback_corr" day center-of-mass.
    
    Returns:
        pd.DataFrame: shrunken ewma covariance matrix.
    """

    n = returns.shape[1]

    # Calculate correlation estimates    
    corr = returns.ewm(span=lookback_corr).corr().droplevel(0).iloc[-n:]
    
    # Calculate volatility estimates
    vols = returns_non_3d.ewm(span=lookback_vol).std().iloc[-1]

    # Shrink off-diagonal correlations
    shrunken_corr = corr * (1 - shrinkage)
    np.fill_diagonal(shrunken_corr.values, 1)
    
    # Compute covariance matrix
    cov = corr_to_cov(corr=shrunken_corr, vols=vols)
    
    if cov.isnull().values.any():
        print(cov)
        raise ValueError("Covariance matrix contains missing values")
    
    return cov


# -------------------------------- Utilities --------------------------------

risk_models = { "empirical_cov" : empirical_cov,
                "shrunken_empirical_cov" : shrunken_empirical_cov,
                "empirical_trailing_cov" : empirical_trailing_cov, 
                "shrunken_empirical_trailing_cov" : shrunken_empirical_trailing_cov,
                "ewma_cov" : ewma_cov,
                "shrunken_ewma_cov" : shrunken_ewma_cov
              }

def preprocess_risk_model_returns(returns: pd.DataFrame, method: str, min_samples: int, apply_rolling_returns: bool, strict_min_samples: bool,**kwargs):
    """ 
    Preprocess, clean, and inspect returns for risk all models.

    Args:
        returns (pd.DataFrame): historical returns.
        method (str, optional): covariance risk model. Defaults to "empirical_cov".
        min_samples (int): minimum number of observations per column to form covariance matrix.
        apply_rolling_returns (bool): if true, applies rolling mean to mitigate effects of asynchronous trading in global asset classes.
        strict_min_samples (bool): if true, drop columns with fewer than ("lookback_cov" | "lookback_corr" | "lookback_vol") samples.
    """
    
    # Risk model lookback parameters
    lookback_params = ["lookback_cov", "lookback_corr"]

    # Risk models that leverage vol estimation
    vol_dependent_risk_models = ["shrunken_empirical_cov", "shrunken_empirical_trailing_cov","ewma_cov", "shrunken_ewma_cov"]

    # Store original, non-rolling returns
    original_returns = returns

    if apply_rolling_returns:
        
        # Calculate rolling 3-day mean returns (mitigates effects of asynchronous trading in global asset classes)
        returns = returns.rolling(window=ROLLING_WINDOW).mean()

        # Update lookback parameters if they exist in kwargs -- lookback periods must be subtracted by 2 after applying rolling(3).mean()
        min_samples -= 2
        for param in lookback_params:
            # If specified risk model uses current param 
            if param in kwargs:
                kwargs[param] -= 2

    # Handle NaNs
    returns = utils.drop_columns_with_nan(returns)

    # Drop securities with insufficient data for estimatation -- n_samples < (min_samples | lookback_cov | lookback_corr)
    returns = utils.drop_columns_below_min_length(df=returns, min_samples=min_samples)

    # If user wants strict adherence to min_sample params (i.e., min length for each param like lookback_cov, lookback_corr, etc.)
    if strict_min_samples:
        for param in lookback_params:
            # If specified risk model uses current param
            if param in kwargs:
                utils.drop_columns_below_min_length(df=returns, min_samples=kwargs[param])   

    # Check if the risk model requires estimation of vol which would require "returns_non_3d" (vol is indifferent to asynchronus trading)
    if method in vol_dependent_risk_models:
        
        # Get non-rolling, original returns
        returns_non_3d = original_returns[returns.columns] 
        
        # If user wants strict adherence to min_sample params
        if strict_min_samples:
            # Drop securities with insufficient data for estimatation if "lookback_vol" is specified 
            returns_non_3d = utils.drop_columns_below_min_length(df=returns_non_3d, min_samples=kwargs.get("lookback_vol", 0))
        
        # Ensure columns are matching
        columns = returns_non_3d.columns.intersection(returns.columns)
        returns = returns[columns]
        returns_non_3d = returns_non_3d[columns]

        # Update kwargs
        kwargs["returns_non_3d"] = returns_non_3d

    return (returns, kwargs)

def corr_to_cov(corr: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    """ Convert a correlation matrix to a covariance matrix using the given volatility (std) vector.

    Args:
        corr (pd.DataFrame): correlation matrix.
        vols (pd.Series): volatility vector.

    Returns:
        pd.DataFrame: covariance matrix converted from correlation matrix and standard deviations.
    """

    # Element-wise product of vols (e.g., vol1*vol1, vol1*vol2, vol2*vol1, vol2*vol2)
    # Intuitively, compute all possible pairwise products between the elements of two vectors
    vol_product = np.outer(vols, vols)

    # Multiply vol1 * vol2 * corr1,2 for the off-diagonals
    cov = corr * vol_product

    return cov

def cov_to_corr(cov: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    """ Convert a covariance matrix to a correlation matrix using the given volatility (std) vector.

    Args:
        cov (pd.DataFrame): covariance matrix.
        vols (pd.Series): volatility vector.

    Returns:
        pd.DataFrame: correlation matrix converted from correlation matrix and standard deviations.
    """

    # Element-wise product of vols (e.g., vol1*vol1, vol1*vol2, vol2*vol1, vol2*vol2)
    # Intuitively, compute all possible pairwise products between the elements of two vectors
    vol_product = np.outer(vols, vols)

    # Divide cov1,2 / vol1 * vol2
    corr = cov / vol_product 

    return corr