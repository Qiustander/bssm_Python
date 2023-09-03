import tensorflow as tf
import numpy as np
from Models.ssm_nlg import NonlinearSSM
"""
Univariate and Multivariate Stochastic Volativity model
"""

def gen_data(testcase='multivariate',
             num_timesteps=200, state_dim=1,
             observed_dim=1):
    pass