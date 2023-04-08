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
# - result_table: calculates result table (face value, market value, etc.)

# IMPORTS
import pandas as pd
import numpy as np
from scipy.stats import norm

class CLOModel():
    def __init__(self, **kwargs):
        self.setup(**kwargs) # set up parameters, by running function setup() on initialization
        self.load_default() # load default rates
        
    def setup(self, **kwargs):
        # (1) Brownian Motion
        self.V0 = 100 # asset value period 0
        self.rf = 0.004 # risk free interest rate
        self.c = 0.05 # risk premium parameter
        self.sigma_m = 0.8*0.14 # market level variance (0.8 stems from beta)
        self.sigma_j = 0.25 # firm level variance
        self.sigma = (self.sigma_m ** 2 + self.sigma_j ** 2) ** 0.5 # total variance parameter
        self.T = 5 # time to maturity
        self.m = 100 # number of steps (granularity of process)
        self.n = 10000 # number of simulations
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
        return pd.read_excel('data\default_rates_SP.xlsx', sheet_name='adjust', header=2, index_col='Rating')
        
    def GBM_fig(self):
        '''Geometric Brownian Motion for illustration purpose
        RETURNS:
        Numpy array of dimensions (1 + number of simulations, total steps)
        '''
        # see: https://quantpy.com.au/stochastic-calculus/simulating-geometric-brownian-motion-gbm-in-python/

        # (1) define calculation parts
        dt = self.T / self.m # calculate each step (total time / frequency)
        drift = (self.rf + self.c) - 0.5 * self.sigma ** 2
        
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
    
    def GBM(self, risk_neutral=False):
        '''Geometric Brownian Motion for CLO model        
        RETURNS:
        Numpy array of dimensions (time to maturity + 1, # of simulations, # of loans)
        '''
        
        # (1) define calculation parts
        if risk_neutral: drift = self.rf - 0.5 * self.sigma ** 2
        else: drift = (self.rf + self.c) - 0.5 * self.sigma ** 2

        # (2) draw and prepare array
        np.random.seed(2020) # set seed
        W = np.random.normal(loc=0, scale=1, size=(self.T+1, self.n, self.j+1)) # draw normal distribution

        # (3) calculate increments and take sum
        diff = self.sigma_m * W[:,:,0].reshape((self.T+1, self.n, 1)) + self.sigma_j * W[:,:,1:]
        incr = drift + diff
        incr[0,:,:] = 0 # period t=1 has no drift/diffusion yet
        
        return self.V0 * np.exp(incr.cumsum(axis=0))
    
    def SPV_value(self, risk_neutral=False):
        '''SPV terminal values
        '''
        # (1) draw asset paths
        if risk_neutral: V = self.GBM(risk_neutral=True)[-1,:,:]
        else: V = self.GBM()[-1,:,:] # take only terminal values
        
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
                                 - ( (self.rf + self.c) - 0.5 * self.sigma ** 2 ) * self.T )
    
    def result_table(self):
        '''Creates result table
        RETURNS:
        DataFrame with model simulation results
        '''
        # (1) create DataFrame
        df = self.default
        df = df.loc[:'B', [self.T]] # select only maturity T
        df.rename(columns={self.T:'default probability'}, inplace=True)
        
        # (2) add face value column
        SPV_values = self.SPV_value()
        df['aggregate face value'] = np.quantile(SPV_values, df['default probability'].values/100) # quantile of sim. dist.
        
        # (3) add market value column
        SPV_values_Q = self.SPV_value(risk_neutral=True) # risk neutral SPV values
        
        for k in df.index: # for each rating
            Bk = np.quantile(SPV_values, df.loc[k, 'default probability']/100) # tranche value
            df.loc[k, 'aggregate market value'] = np.minimum(SPV_values_Q, Bk).mean() * np.exp(-self.rf * self.T)
            
            
        # (4) tranching to find face value and market value
        df['face value'] = df['aggregate face value'] - df['aggregate face value'].shift(1)
        df.loc['AAA', 'face value'] = df.loc['AAA', 'aggregate face value']
        df['market value'] = df['aggregate market value'] - df['aggregate market value'].shift(1)
        df.loc['AAA', 'market value'] = df.loc['AAA', 'aggregate market value']
        
        # (5) kurs column
        df['kurs'] = df['market value'] / df['face value'] * 100
        
        # (6) yield and spread columns
        df['afkast'] = 1 / self.T * np.log(df['face value'] / df['market value']) * 100
        df['spread'] = df['afkast'] - self.rf * 100
          
        return df  