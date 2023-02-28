######################
# model_functions.py #
######################

# This module contains the model functions class which contains the following functions:

# - setup (called by init): sets up model parameters

class clo_model():
    def __init__(self,**kwargs):
        self.setup(**kwargs) # set up parameters, by running function setup() on initialization
        
    def setup(self,**kwargs):
        #(1) Model parameters
        self.parameter = 1.0

        #(2) Update baseline parameters using keywords
        for key,val in kwargs.items():
            setattr(self,key,val)