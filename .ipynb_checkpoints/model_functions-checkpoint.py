######################
# model_functions.py #
######################

# This module contains the CLOModel class which contains the following functions:

# - setup (called by init): sets up model parameters
# - load_default: loads Cumulative Default Rates DataFrame
# - GBM_fig: draws an array of underlying asset value paths for the purpose of plotting
# - GBM: draws an array of underlying asset value paths for the CLO model
# - SPV_value: simulates assets and calculates SPV values
# - face_value: calculates the face value of a loan, given rating and maturity

# IMPORTS
import pandas as pd
import numpy as np
from scipy.stats import norm

class CLOModel():
    def __init__(self,**kwargs):
        self.setup(**kwargs) # set up parameters, by running function setup() on initialization
        self.load_default() # load default rates
        
    def setup(self,**kwargs):
        # (1) Brownian Motion
        self.V0 = 100 # asset value period 0
        self.mu = 0.091 # drift coefficient
        self.sigma_m = 0.8*0.14 # market level variance (0.8 stems from beta)
        self.sigma_j = 0.25 # firm level variance
        self.sigma = (self.sigma_m ** 2 + self.sigma_j ** 2) ** 0.5 # total variance parameter
        self.T = 5 # time to maturity
        self.m = 100 # number of steps (granularity of process)
        self.n = 1000 # number of simulations
        self.j = 100 # number of loans
        
        # (2) SPV cash flow and value calculations
        self.rating = 'B' # rating for each loan
        self.default = self.load_default()
        self.B = self.face_value() # call face value function to calculate
        
        # Update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val)

    def load_default(self):
        '''Loads S&P Cumulative Default Rates for 1981-2021 as a dataframe'''
        return pd.read_excel('data\default_rates_SP.xlsx', header = 2, index_col = "Rating")
        
    def GBM_fig(self):
        '''Geometric Brownian Motion for illustration purpose
        RETURNS:
        Numpy array of dimensions (1 + number of simulations, total steps)
        '''
        # see: https://quantpy.com.au/stochastic-calculus/simulating-geometric-brownian-motion-gbm-in-python/
        
        # (1) define calculation parts
        dt = self.T / self.m # calculate each step (total time / frequency)
        drift = self.mu - 0.5 * self.sigma ** 2
        
        # (2) draw and prepare array
        np.random.seed(50) # set seed
        W = np.random.normal(loc=0, scale=np.sqrt(dt), size=(self.n, self.m)) # draw normal dist
        Vt = np.exp(drift * dt + self.sigma * np.transpose(W))
        Vt = np.vstack([np.ones(self.n), Vt])
    
        #Might want to do this to save the normal draws too:
        # if return_normal:
            # return self.S0 * St.cumprod(axis=0), normal
            
        # use cumulative product (over rows) to calculate simulation paths, and multiply by initial value V0
        return self.V0 * Vt.cumprod(axis=0) # axis=0 to calculate over rows
    
    def GBM(self):
        '''Geometric Brownian Motion for CLO model        
        RETURNS:
        Numpy array of dimensions (time to maturity + 1, # of simulations, # of loans)
        '''
        # (1) define calculation parts
        drift = self.mu - 0.5 * self.sigma ** 2

        # (2) draw and prepare array
        np.random.seed(50) # set seed
        W = np.random.normal(loc=0, scale=1, size=(self.T+1, self.n, self.j+1)) # draw normal distribution

        # (3) calculate increments and take sum
        diff = self.sigma_m * W[:,:,0].reshape((self.T+1, self.n, 1)) + self.sigma_j * W[:,:,1:]
        incr = drift + diff
        incr[0,:,:] = 0 # period t=1 has no drift/diffusion yet
        
        return self.V0 * np.exp(incr.cumsum(axis=0))
    
    def SPV_value(self):
        '''SPV terminal values
        '''
        # (1) draw asset paths
        V = self.GBM()[-1,:,:] # take only terminal values
        
        # (2) calculate minimum and take sum
        CF = np.minimum(V, self.B)
        CF_sum = np.sum(CF, axis=1) # take sum over firms j = 1, ..., J
        
        return np.sort(CF_sum)
    
    def face_value(self):
        '''Face value from cumulative default probability table
        RETURNS:
        Face value B
        '''
        def_prob = self.default.loc[self.rating, self.T]/100 # cumulative default probability from table        
        
        return self.V0 / np.exp( - norm.ppf(def_prob) * self.sigma * np.sqrt(self.T)
                                 - ( self.mu - 0.5 * self.sigma ** 2 ) * self.T )
        
        
        