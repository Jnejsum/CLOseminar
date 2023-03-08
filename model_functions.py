######################
# model_functions.py #
######################

# This module contains the model functions class which contains the following functions:

# - setup (called by init): sets up model parameters

# IMPORTS
import pandas as pd

class CLOModel():
    def __init__(self,**kwargs):
        self.setup(**kwargs) # set up parameters, by running function setup() on initialization
        self.load_default() # load default rates
        
    def setup(self,**kwargs):
        #(1) Model parameters
        self.parameter = 1.0
        
        #(2) Update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val)

    def load_default(self):
        '''Loads S&P Cumulative Default Rates for 1981-2021 as a dataframe'''
        self.default = pd.read_excel('data\default_rates_SP.xlsx', header = 2, index_col = "Rating")
        
    def GBM(self, ):
        '''Geometric Brownian Motion'''
        