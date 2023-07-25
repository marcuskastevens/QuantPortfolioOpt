'''
Library for portfolio optimization functions.

Supported Optimization Algorithms:
1) Unconstrained Max Sharpe Ratio (Closed-Form Lagrangian) 
2) Max Sharpe Ratio (MVO)
3) Mean-Variance (Risk Averse MVO)
4) Mean-Variance + Turnover Constraints (Risk + Turnover Averse MVO)
5) Risk Parity (Equal Variance Contribution)
6) Dollar Risk Parity (Equal Dollar Contribution of Variance Risk)
7) ATR Risk Parity (Equal Dollar Contribution of "True Range" Risk)

Future Optimization Algorithms:
1) Min Variance
2) Max Ucler Index / Martin Ratio
3) Max Calmar Ratio
4) HRP 
5) NCO
6) Min Entropic VaR / Minimum Kurtosis Portfolios
7) Max Sortino 
8) Max Return + Risk Aversion
9) Max Return + Risk Aversion + Turnover Aversion
'''


from scipy.optimize import minimize as opt
from quant_tools import risk_analysis as ra, performance_analysis as pt, data_preprocessing as dp
from quant_tools.beta import obj_functions
from scipy.optimize import Bounds
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np

# -------------------------------- Optimization Functions --------------------------------

def unconstrained_max_sharpe_mvo(returns: pd.DataFrame, expected_returns: pd.Series, cov: pd.DataFrame, verbose: bool = False) -> pd.Series:
    """ Implements MVO closed-form solution for maximizing Sharpe Ratio.

    Args:
        returns (pd.DataFrame): historical returns matrix.
        expected_returns (pd.Series): expected returns vector.
        verbose (bool): if true, print relevant optimization information.

    Returns:
        pd.Series: optimized weights vector.
    """
    # Get E[SR] and correlation matrix
    # expected_sr = returns.mean()/returns.std()*252**.5 # returns.mean()
    # inverse_corr = np.linalg.inv(returns.corr()) # np.linalg.inv(returns.cov()).round(4) 

    # Get E[r] and covariance matrix
    invserse_cov = np.linalg.inv(cov)

    # Get MVO Vol Weights and Convert them to Standard Portfolio Weights
    numerator = np.dot(invserse_cov, expected_returns)
    denominator = np.sum(numerator)
    w = numerator / denominator
    w = pd.Series(w, index=returns.columns)

    # Print relevant allocation information
    if verbose:
        print('Target Vol: ')
        print(np.sqrt(np.dot(np.dot(w.T, cov), w)))
        
        ls_ratio = np.abs(w[w>0].sum() / w[w<0].sum())
        print(f'Long-Short Ratio: {ls_ratio}')
        print(f'Leverage: {w.abs().sum()}')
        print(f'Sum of Vol Weights: {w.sum().round(4)}') 
        mvo_sr = obj_functions.sharpe_ratio_obj(w, expected_returns, cov, neg=False)
        print(f'Target Portfolio Sharpe Ratio: {mvo_sr}')   
    
    return w


def max_sharpe_mvo(returns: pd.DataFrame, expected_returns: pd.DataFrame, cov: pd.DataFrame, long_only = False, vol_target = .01, max_position_weight = .2, max_leverage = 1, market_bias = 0, verbose=False) -> pd.Series:
    """ Constrained or unconstrained Mean Variance Optimization. This leverages convex optimization to identify local minima which serve to minimize an objective function.
        In the context of portfolio optimization, our objective function is the negative portfolio SR. 

    Args:
        returns (pd.DataFrame): expanding historical returns of specified asset universe
        expected_returns (pd.DataFrame): expected returns across specified asset universe, normally computed via statistical model
        vol_target (float, optional): targeted ex-ante volatilty based on covariance matrix. Defaults to .10.

    Returns:
        pd.Series: optimized weights vector.
    """

    # Match tickers across expected returns and historical returns
    expected_returns = expected_returns.dropna()

    n = expected_returns.shape[0]

    if n > 0:
        
        # Initial guess is naive 1/n portfolio
        initial_guess = np.array([1 / n] * n)

        if long_only: 
            bounds = Bounds(0, max_position_weight)
        else:
            # Set max allocation per security
            bounds = Bounds(-max_position_weight, max_position_weight)

        constraints =  [# Target volatility
                        #{"type": "eq", "fun": lambda w: np.sqrt(np.dot(np.dot(w.T, cov), w)) - vol_target},
                        
                        # Ensure dollar neutral portfolio (or alternatively specified market bias)
                        {"type": "eq", "fun": lambda w: np.sum(w) - market_bias},

                        # Target Leverage (Net Exposure)
                        # {"type": "ineq", "fun": lambda w: np.sum(np.abs(w)) - (notional_exposure - .01)}, # 0.99 <= weights.sum
                        # {"type": "ineq", "fun": lambda w: (notional_exposure + .01) - np.sum(np.abs(w))}, # 1.01 >= weights.sum

                        ]

        w = opt(obj_functions.sharpe_ratio_obj, 
                initial_guess,
                args=(cov, expected_returns), 
                method='SLSQP', 
                bounds = bounds,
                constraints=constraints)['x']
        
        w = pd.Series(w, index=expected_returns.index)

        
        # Target risk
        ex_ante_vol = np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252)
        vol_scalar = vol_target / ex_ante_vol
        w *= vol_scalar

        # Handle leverage constraints
        leverage = np.sum(w)
        if leverage > max_leverage:
            leverage_scalar = max_leverage / np.sum(w)
            w *= leverage_scalar
              
        # Print relevant allocation information
        if verbose:
            print('Target Vol: ')
                      
            ls_ratio = np.abs(w[w>0].sum() / w[w<0].sum())
            print(f'Long-Short Ratio: {ls_ratio}')
            print(f'Leverage: {w.abs().sum()}')
            mvo_sr = obj_functions.sharpe_ratio_obj(w, expected_returns, cov, neg=False)
            print(f'Target Portfolio Sharpe Ratio: {mvo_sr}')
        
        return w
    
    return None

# Risk Averse MVO
def risk_averse_mvo(returns: pd.DataFrame, expected_returns: pd.DataFrame, cov: pd.DataFrame, gamma: float = .5, long_only = False, vol_target = .01, max_position_weight = .2, max_leverage = 1, market_bias = 0, verbose=False) -> pd.Series:

    n = expected_returns.shape[0]

    if n > 0:
        
        # Initial guess is naive 1/n portfolio
        initial_guess = np.array([1 / n] * n)

        if long_only: 
            bounds = Bounds(0, max_position_weight)
        else:
            # Set max allocation per security
            bounds = Bounds(-max_position_weight, max_position_weight)

        constraints =  [# Dollar neutral portfolio (or alternatively specified market bias)
                        {"type": "eq", "fun": lambda w: np.sum(w) - market_bias},
                       ]
        
        w = opt(obj_functions.mvo_w_risk_aversion_obj, 
                initial_guess,
                args=(cov, expected_returns, gamma), 
                method='SLSQP', 
                bounds = bounds,
                constraints=constraints)['x']
        
        w = pd.Series(w, index=expected_returns.index)
        
        # Target risk
        ex_ante_vol = np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252)
        vol_scalar = vol_target / ex_ante_vol
        w *= vol_scalar

        # Handle leverage constraints
        leverage = np.sum(w)
        if leverage > max_leverage:
            leverage_scalar = max_leverage / np.sum(w)
            w *= leverage_scalar

        print(np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252))

        return w
    
    return None
        
        

# Risk & Turnover Averse MVO
def risk_turnover_averse_mvo(returns: pd.DataFrame, expected_returns: pd.DataFrame, cov: pd.DataFrame, w1: pd.Series, gamma: float = .5, delta: float = .5, long_only = False, vol_target = .01, max_position_weight = .2, max_leverage = 1, market_bias = 0, verbose=False) -> pd.Series:

    n = expected_returns.shape[0]

    if n > 0:
        
        # Initial guess is naive 1/n portfolio (stable optimization convergence)
                # rand_w = np.random.rand(n) 
                # rand_w = rand_w / np.sum(rand_w)
                # initial_guess = rand_w
        initial_guess = np.array([1 / n] * n)

        # Set max allocation per instrument
        if long_only: 
            bounds = Bounds(0, max_position_weight)
        else:
            bounds = Bounds(-max_position_weight, max_position_weight)

        constraints =  [# Dollar neutral portfolio (or alternatively specified market bias)
                        {"type": "eq", "fun": lambda w: np.sum(w) - market_bias},
                       ]
        
        w = opt(obj_functions.mvo_w_risk_turnover_aversion_obj, 
                initial_guess,
                args=(w1, cov, expected_returns, gamma, delta), 
                method='SLSQP', 
                bounds = bounds,
                constraints=constraints)['x']
        
        w = pd.Series(w, index=expected_returns.index)
        
        # Target risk
        ex_ante_vol = np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252)
        vol_scalar = vol_target / ex_ante_vol
        w *= vol_scalar

        # Handle leverage constraints
        leverage = np.sum(w)
        if leverage > max_leverage:
            leverage_scalar = max_leverage / np.sum(w)
            w *= leverage_scalar

        return w
    
    return None



def naive_risk_parity(returns: pd.DataFrame, cov: pd.DataFrame, vol_target: float = .1, max_leverage: float = 1) -> pd.Series:
    """ Naive Risk Parity portfolio construction algorithm to get equal risk 
        contribution (variance) portfolio weights assuming assets are uncorrelated.

        Closed-form solution: w = σ^-1 / I.T.dot(σ^-1)

    Args:
        cov (pd.DataFrame): estimated covariance matrix of returns.

    Returns:
        pd.Series: niave risk parity portfolio weights.
    """

    # Get labeled variance vector
    variance = pd.Series(np.diag(cov), index=cov.index)
    
    # Inverse variance
    inv_variance = 1 / variance

    # Inverse variance scaled to create portfolio weights
    w = inv_variance / np.sum(inv_variance)

    # Target risk
    ex_ante_vol = np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252)
    vol_scalar = vol_target / ex_ante_vol
    w *= vol_scalar

    # Handle leverage constraints
    leverage = np.sum(w)
    if leverage > max_leverage:
        leverage_scalar = max_leverage / np.sum(w)
        w *= leverage_scalar

    return w

def vanilla_risk_parity(returns: pd.DataFrame, cov: pd.DataFrame, vol_target: float = 0.1, max_leverage: float = 1, verbose: bool = False) -> pd.Series:
    """ Generalized Risk Parity portfolio construction algorithm to 
        get equal risk contribution (variance) portfolio weights.

        Minimize: Σ(rc - rc_equal)^2
        Subject to: 
            1) w >= 0
            2) I.T.dot(w) = 1

    Args:
        returns (pd.DataFrame): portfolio constituents' historical returns.
        cov (pd.DataFrame): estimated covariance matrix of returns.
        vol_target (float): annualized target volatility.
        max_leverage (float): max notional exposure of portfolio.
        verbose (bool): if true, print relevant portfolio statistics.

    Returns:
        pd.Series: risk parity portfolio weights.
    """

    n = len(returns.columns)

    initial_guess = pd.Series(np.array([1/n] * n), index=returns.columns)

    # Long-only (w > 0)
    bounds = Bounds(0, np.inf)

    constraints =   [# Portfolio weights sum to 100%
                    {"type": "eq", "fun": lambda w: w.sum() - 1}
                    ]

    # Get risk parity weights
    w = opt(obj_functions.vanilla_risk_parity_obj, 
                initial_guess, 
                args=(cov), 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints)['x']

    w = pd.Series(w, index=cov.index)

    # Target risk
    ex_ante_vol = np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252)
    vol_scalar = vol_target / ex_ante_vol
    w *= vol_scalar

    if verbose:
        print(f"initial ex_ante_vol - {ex_ante_vol}")
        print(f"vol scaled ex_ante_vol - {np.sqrt(w.T.dot(cov).dot(w))*np.sqrt(252)}")
    
    # Handle leverage constraints
    leverage = np.sum(w)
    if leverage > max_leverage:
        leverage_scalar = max_leverage / np.sum(w)
        w *= leverage_scalar
    
    if verbose:
        print(f'post_constraint leverage - {w.sum()}')
        print(f'ex_ante vol {np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252)}', w.T.dot(cov).dot(w)*252)   

    return w

def expected_returns_risk_parity(returns: pd.DataFrame, cov: pd.DataFrame, expected_returns: pd.Series, gamma: float = .5, vol_target: float = 0.1, max_leverage: float = 1, long_only=True, verbose: bool = False) -> pd.Series:
    """ Enhanced Risk Parity portfolio construction algorithm to get equal risk contribution (variance) 
        portfolio weights while accounting for expected returns.

        Minimize: Σ(rc - rc_equal)^2 - λ * w.T.dot(µ)
        Subject to: 
            1) w >= 0
            2) I.T.dot(w) = 1 or I.T.dot(w) = 0

    Args:
        returns (pd.DataFrame): portfolio constituents' historical returns.
        cov (pd.DataFrame): estimated covariance matrix of returns.
        expected_returns (pd.Series): vector of expected returns.
        gamma (float): return regularization parameter (importance of expected returns).

    Returns:
        pd.Series: risk parity portfolio weights.
    """

    n = len(returns.columns)

    initial_guess = pd.Series(np.array([1/n] * n), index=returns.columns)

    # Long-only (w > 0)
    if long_only:
        bounds = Bounds(0, np.inf)
        constraints =   [# Portfolio weights sum to 100%
                        {"type": "eq", "fun": lambda w: w.sum() - 1}
                        ]
    else:
        bounds = Bounds(-np.inf, np.inf)
        constraints =   [# Dollar neutral
                        {"type": "eq", "fun": lambda w: w.sum() - 0}
                        ]

    # Get risk parity weights
    w = opt(obj_functions.expected_returns_risk_parity_obj, 
                initial_guess, 
                args=(cov, expected_returns, gamma), 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints)['x']

    w = pd.Series(w, index=cov.index)

    # Target risk
    ex_ante_vol = np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252)
    vol_scalar = vol_target / ex_ante_vol
    w *= vol_scalar

    if verbose:
        print(f"initial ex_ante_vol - {ex_ante_vol}")
        print(f"vol scaled ex_ante_vol - {np.sqrt(w.T.dot(cov).dot(w))*np.sqrt(252)}")
    
    # Handle leverage constraints
    leverage = np.sum(w)
    if leverage > max_leverage:
        leverage_scalar = max_leverage / np.sum(w)
        w *= leverage_scalar
    
    if verbose:
        print(f'post_constraint leverage - {w.sum()}')
        print(f'ex_ante vol {np.sqrt(w.T.dot(cov).dot(w)) * np.sqrt(252)}', w.T.dot(cov).dot(w)*252)  

    return w

def dollar_risk_parity(prices: pd.DataFrame, cov: pd.DataFrame, target_risk = 0.001, portfoio_value = 100000, long_only=False, verbose=False) -> pd.Series:
    """ Inspired by CTA and Trend Follwers' risk management practices, the Dollar Risk Parity optimization function
        targets "target_risk" percent risk per position (equal risk contribution) which represents 1 SD of the 
        underlying instrument's price movement. Here, the covariance and variance of each asset defines its risk to account for
        correlation structures and mitiage over exposure to a single risk factor.
               
        Positions represent how many shares to purchase for a 1SD move in the underlying instrument 
        to represent "target_risk" percent loss in "portfolio_value". This follows the risk management practices of 
        select portfolio managers that target equal risk allocation to each trade/position, but leverages stop-losses 
        (e.g., 1SD stop) to limit exogenous risk exposure.

        This follows the intuition behind traditional equal variance contribution risk parity, but defines risk as dollar value
        at risk (e.g., stop loss set at 1SD) instead of purely variance. 

    Args:
        prices (pd.DataFrame): _description_
        target_risk (float, optional): _description_. Defaults to 0.005.
        portfoio_value (int, optional): _description_. Defaults to 100000.

    Returns:
        pd.Series: _description_
    """
    n = len(prices.columns)

    initial_guess = pd.Series(np.array([1/n] * n), index=prices.columns)

    constraints =   [# Ensure notional exposure < portfolio_value (i.e., no leverage)
                    {"type": "ineq", "fun": lambda n_units: portfoio_value - (n_units*prices.iloc[-1]).sum()}
                    ]

    # Long-Only or L/S
    if long_only:
        bounds = Bounds(0, np.inf)
    else:
        bounds = Bounds(-np.inf, np.inf)

    # Get dollar risk parity weights
    n_units = opt(obj_functions.dollar_risk_parity_obj, 
            initial_guess, 
            bounds=bounds,
            args=(cov),
            method='SLSQP',
            constraints=constraints)['x']

    # Assign units to position labels
    n_units = pd.Series(n_units, index=cov.index)

    # Compute target portfolio risk (dollar risk at 1SD move)
    target_portfolio_risk = portfoio_value*n*target_risk

    # Get ex-ante dollar risk at 1SD
    ex_ante_dollar_vol = np.sqrt(n_units.T.dot(cov).dot(n_units))
    
    # Compute risk scalar to ensure target risk is acheived
    risk_scalar = target_portfolio_risk / ex_ante_dollar_vol

    # Multiply optimized positions by risk scalar to target vol
    n_units *= risk_scalar

    # Re-compute ex-ante dollar risk at 1SD
    ex_ante_scaled_dollar_vol = np.sqrt(n_units.T.dot(cov).dot(n_units))

    if verbose:
        print(f"Target Portfolio Risk: {target_portfolio_risk}")
        print(f"Ex-Ante Portfolio Risk: {ex_ante_scaled_dollar_vol}")
        print(f"Ex-Ante Dollar Risk (1SD) Contributions: \n{(n_units.T.dot(cov) * n_units)**.5}" )

    return n_units


def atr_risk_parity(prices: pd.DataFrame, true_ranges: pd.DataFrame, cov: pd.DataFrame, lookback_window=20, target_risk = 0.001, portfoio_value = 100000, long_only=False, verbose=False) -> pd.Series:
    """ Inspired by CTA and Trend Follwers' risk management practices, the Dollar Risk Parity optimization function
        targets "target_risk" percent risk per position (equal risk contribution) which represents 1 SD of the 
        underlying instrument's price movement. Here, the covariance and variance of each asset defines its risk to account for
        correlation structures and mitiage over exposure to a single risk factor.
               
        Positions represent how many shares to purchase for a 1SD move in the underlying instrument 
        to represent "target_risk" percent loss in "portfolio_value". This follows the risk management practices of 
        select portfolio managers that target equal risk allocation to each trade/position, but leverages stop-losses 
        (e.g., 1SD stop) to limit exogenous risk exposure.

        This follows the intuition behind traditional equal variance contribution risk parity, but defines risk as dollar value
        at risk (e.g., stop loss set at 1SD) instead of purely variance. 

    Args:
        prices (pd.DataFrame): _description_
        target_risk (float, optional): _description_. Defaults to 0.005.
        portfoio_value (int, optional): _description_. Defaults to 100000.

    Returns:
        pd.Series: _description_
    """

    n = len(prices.columns)

    # Covariance of prices
    # cov = dp.true_range_covariance(true_ranges, lookback_window=lookback_window)

    initial_guess = pd.Series(np.array([1/n] * n), index=prices.columns)

    constraints =   [# Notional exposure < portfolio_value (i.e., no leverage)
                    {"type": "ineq", "fun": lambda n_units: portfoio_value - prices.iloc[-1].dot(n_units)},
                    # Target Risk Level - Doesn't Work - Scale Risk After Opt Instead
                    # {"type": "eq", "fun": lambda n_units: np.sqrt(n_units.T.dot(cov).dot(n_units)) - portfoio_value*n*target_risk}
                    ]

    # Long-Only or L/S
    if long_only:
        bounds = Bounds(0, np.inf)
        # bounds = Bounds(0, 10000000000)
    else:
        bounds = Bounds(-np.inf, np.inf)
        # bounds = Bounds(-10000000000, 10000000000)
    
    # Get dollar risk parity weights
    n_units = opt(obj_functions.atr_risk_parity_obj, 
            initial_guess, 
            bounds=bounds,
            args=(cov),
            method='SLSQP',
            constraints=constraints)['x']

     # Assign units to position labels
    n_units = pd.Series(n_units, index=cov.index)

    # Compute target portfolio risk (dollar risk at 1SD move)
    target_portfolio_risk = portfoio_value*n*target_risk

    # Get ex-ante dollar risk at 1SD
    ex_ante_dollar_vol = np.sqrt(n_units.T.dot(cov).dot(n_units))
    
    # Compute risk scalar to ensure target risk is acheived
    risk_scalar = target_portfolio_risk / ex_ante_dollar_vol

    # Multiply optimized positions by risk scalar to target vol
    n_units *= risk_scalar

    # Control for Leverage - alternative is to impose this in the convex optimization constraints
    if prices.iloc[-1].dot(n_units) > portfoio_value:
        print('Leverage Constraint Used')
        leverage_scalar = portfoio_value / prices.iloc[-1].dot(n_units)
        n_units *= leverage_scalar
    
    # Re-compute ex-ante dollar risk at 1SD
    ex_ante_true_range_risk = np.sqrt(n_units.T.dot(cov).dot(n_units))

    if verbose: 
        print(f"Target Portfolio Risk: {portfoio_value*n*target_risk}")
        print(f"Ex-Ante True Range Dollar Risk: {ex_ante_true_range_risk}")
        print(f"Ex-Ante True Range Risk Contributions: \n{(n_units.T.dot(cov) *  n_units)**.5}")

    return n_units


# -------------------------------- Utils --------------------------------

def update_args(args: dict, returns: pd.DataFrame, w: dict, opt_method: str, expected_return_method=None):

    # Turnover constrained optimization functions
    turnover_constrained_opts = ["risk_turnover_averse_mvo"]

    # Get specified optimization algorithm
    optimization_algo = optimization_algo_map[opt_method]

    # Update returns
    args['returns'] = returns

    # Update expected returns if applicable
    if expected_return_method:
        args['expected_returns'] = returns.tail(126).mean() # expected_return_map[expected_return_method](returns) -- implement later
        # args = optimization_args_map[opt_method](returns, args=args, expected_return_method=expected_return_method)

    # If turnover constrained opt
    if opt_method in turnover_constrained_opts:

        # Get previous rebal date's portfolio weights
        w1 = pd.DataFrame(w).T.iloc[-1]

        # Ensure tradable assets are compatable
        columns = w1.index.intersection(returns.columns)
                
        # Update args
        args['w1'] = w1[columns]

    # Get opt function args
    supported_args = list(optimization_algo.__code__.co_varnames[:optimization_algo.__code__.co_argcount])

    # Drop unsupported args
    unsupported_args = [key for key in args.keys() if key not in supported_args]
    if len(unsupported_args) > 0: 
        print(f"{unsupported_args} are not supported args in the {opt_method} optimization function!")
        print(f"Supported args are {supported_args}.")
        for key in unsupported_args:
            del args[key]

    # Get required opt function args (slice args for required args)
    required_args = list(optimization_algo.__code__.co_varnames[:optimization_algo.__code__.co_argcount])

    # Check if all required args are defined
    missing_required_args = [key for key in required_args if key not in args.keys()]
    if len(missing_required_args) > 0:
        raise TypeError(f"{opt_method} optimization function missing required argument(s):\n{missing_required_args}")
    
    return args

# Hash map between optimization methods and their respective functions
optimization_algo_map = {   "Unconstrained Max Sharpe Ratio" : unconstrained_max_sharpe_mvo,
                            "max_sharpe_mvo" : max_sharpe_mvo,
                            "risk_averse_mvo" : risk_averse_mvo,
                            "risk_turnover_averse_mvo" : risk_turnover_averse_mvo,
                            "naive_risk_parity" : naive_risk_parity,
                            "vanilla_risk_parity" : vanilla_risk_parity, 
                            "expected_returns_risk_parity" : expected_returns_risk_parity,
                            "Dollar Risk Parity" : dollar_risk_parity,
                            "ATR Risk Parity" : atr_risk_parity,
                        }

# def max_sharpe_mvo_args(returns: pd.DataFrame, expected_return_method: str, args: tuple):

#     args['returns'] = returns
#     args['expected_returns'] = returns.tail(60).mean() # expected_return_map[expected_return_method](returns) -- implement later

#     return args

# def unconstrained_max_sharpe_mvo_args(returns: pd.DataFrame, expected_return_method: str, args: tuple, **kwargs):

#     args['returns'] = returns
#     args['expected_returns'] = returns.tail(60).mean() # expected_return_map[expected_return_method](returns) -- implement later

#     return args

# def risk_parity_args(returns: pd.DataFrame, args: tuple):

#     args['returns'] = returns

#     return args

# def dollar_risk_parity_args(prices: pd.DataFrame, args: tuple):
    
#     args['prices'] = prices

#     return args

# def atr_risk_parity_args(prices: pd.DataFrame, true_ranges: pd.DataFrame, args: tuple):

#     args['prices'] = prices
#     args['true_ranges'] = true_ranges 

#     return args

# Hash map between optimization methods and their respective functions

# Hash map between optimization methods and their respective update arg functions
# optimization_args_map = {  "Unconstrained Max Sharpe Ratio" : unconstrained_max_sharpe_mvo_args,
#                                 "Max Sharpe Ratio" : max_sharpe_mvo_args,
#                                 "Risk Parity" : risk_parity_args, 
#                                 "Dollar Risk Parity" : dollar_risk_parity_args,
#                                 "ATR Risk Parity" : atr_risk_parity_args,
#                         }