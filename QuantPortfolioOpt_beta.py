'''
Library for walk-forward portfolio optimization.

Current Implementations:
1) Generalized Class for Portfolio Optimization
2) Portfolio Stress Testing Methods (Rebalancing Timing Luck)

'''

from quant_tools import risk_analysis as ra, performance_analysis as pt, data_preprocessing as dp
from quant_tools.beta import opt_functions as opt, cov_functions as risk_models
from scipy.optimize import Bounds
import statsmodels.api as sm
from scipy import stats
import datetime as dt
import pandas as pd
import numpy as np

# -------------------------------- Constants --------------------------------
backtest_start_date = dt.date(2007, 1, 1)

# -------------------------------- Optimization Classes --------------------------------
class walk_forward_portfolio_optimization():
    def __init__(self, returns: pd.DataFrame, opt_args: tuple, cov_args: tuple, rebal_freq: int = 21, opt_method: str = "Max Sharpe Ratio", cov_method: str = "ewma_cov", expected_return_method: str = None):
        
        self.returns = returns     
        self.expected_returns = self.returns.rolling(252).mean() # self.get_expected_returns(self.returns) -- perhaps user passes these into function
        self.rebal_freq = rebal_freq
        self.expected_return_method = expected_return_method
        self.opt_method = opt_method
        self.cov_method = cov_method
        # self.constraints -- implement later
        
        self.optimization_algo = opt.optimization_algo_map[self.opt_method] # -- perhaps create a wrapper function for portfolio opt algos just like i did with risk models

        self.cov_algo = risk_models.risk_matrix

        self.opt_args = opt_args
            
        self.cov_args = cov_args
            
        self.w, self.rebal_w = self.run()
        
    def run(self):
        
        # Empty w matrix for indexing purposes
        empty_w = pd.DataFrame(index=self.returns.index)

        # Hash Map to hold walk-forward optimized weights 
        w = {}

        for date in self.returns.index[::self.rebal_freq]:       
            
            # Get expanding historical returns
            tmp_returns = self.returns.loc[:date]

            try: 
                # Estimate covariance matrix 
                tmp_cov = self.cov_algo(returns=tmp_returns, method=self.cov_method, **self.cov_args)
                self.opt_args.update({"cov" : tmp_cov})  
                 
            except:
                print(f'ERROR - {date.date()}')                
                if date.date() > backtest_start_date:         
                    print(tmp_returns.tail())
                    return tmp_returns
                    self.cov_algo(returns=tmp_returns, method=self.cov_method, **self.cov_args)

                w[date] = pd.Series(index=tmp_returns.columns)
                    
                continue

            # Update portfolio constituents -- due to data errors that cov_algo handled 
            tmp_returns = tmp_returns[tmp_cov.columns]

            # Clean, handle errors, & update args where applicable
            opt.update_args(args=self.opt_args, returns=tmp_returns, w=w, opt_method=self.opt_method, expected_return_method=self.expected_return_method)  

            # Tradable universe size
            n = tmp_returns.shape[1]

            if n > 1:
                # Get optimal weights
                w[date] = self.optimization_algo(**self.opt_args)

            elif n == 1: 
                # 100% weight in single security -- handle short vs. long later
                w[date] = pd.Series({tmp_returns.columns[0] : 1.00})   

            else:
                # No tradable assets on given date
                continue    

        # Save rebal date weights & forward filled weights
        rebal_w = pd.DataFrame(w).T
        w = pd.concat([rebal_w, empty_w], axis=1).ffill()
            
        return w, rebal_w

