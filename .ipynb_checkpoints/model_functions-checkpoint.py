######################
# model_functions.py #
######################

# This module contains the CLOModel class which contains the following functions:

# - setup (called by init): sets up model parameters
# - load_default: loads Cumulative Default Rates DataFrame
# - GBM: draws an array of underlying asset value paths

# IMPORTS
import pandas as pd
import numpy as np

class CLOModel():
    def __init__(self,**kwargs):
        self.setup(**kwargs) # set up parameters, by running function setup() on initialization
        self.load_default() # load default rates
        
    def setup(self,**kwargs):
        #(1) Model parameters
        self.parameter = 1.0 # test parameter
        
        # Brownian Motion
        self.S0 = 100 # asset value period 0
        self.mu = 0.1 # drift coefficient
        self.sigma = 0.3 # volatility
        self.ttm = 5 # time to maturity
        self.freq = 100 # number of steps (granularity of process)
        self.n_sims = 100 # number of simulations
         
        #(2) Update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val)

    def load_default(self):
        '''Loads S&P Cumulative Default Rates for 1981-2021 as a dataframe'''
        self.default = pd.read_excel('data\default_rates_SP.xlsx', header = 2, index_col = "Rating")
        
    def GBM(self):
        '''Geometric Brownian Motion
        RETURNS:
        Numpy array of dimensions (1 + number of simulations, total steps)
        '''
        # see: https://quantpy.com.au/stochastic-calculus/simulating-geometric-brownian-motion-gbm-in-python/
        
        # (1) define calculation parts
        dt = self.ttm / self.freq # calculate each step (total time / frequency)
        drift = self.mu - 0.5 * self.sigma ** 2
        
        # (2) draw and prepare array
        np.random.seed(50) # set seed
        normal = np.random.normal(loc=0, scale=np.sqrt(dt), size=(self.n_sims, self.freq)) # draw normal dist
        St = np.exp(drift * dt + self.sigma * np.transpose(normal))
        St = np.vstack([np.ones(self.n_sims), St])
        
        # use cumulative product (over rows) to calculate simulation paths, and multiply by initial value V0
        
        #Might want to do this to save the normal draws too:
        # if return_normal:
            # return self.S0 * St.cumprod(axis=0), normal
        
        return self.S0 * St.cumprod(axis=0) # axis=0 to calculate over rows
        
        
        
        
        
        
        