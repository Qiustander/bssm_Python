import tensorflow as tf
import numpy as np
from Models.ssm_nlg import NonlinearSSM
"""
Univariate and Multivariate Stochastic Volativity model
  .. math::
    X_0 & \sim N(\mu, \sigma^2/(1-\rho^2)) \\
    X_t & = \mu + \rho(X_{t-1}-\mu) + \sigma U_t, \quad U_t\sim N(0,1) \\
    Y_t|X_t=x_t & \sim N(0, e^{x_t}) \\
"""

def gen_ssm(num_timesteps=200, state_dim=1,
             observed_dim=1):

    state_mtx_noise = .178
    mu = -1.02
    rho = 0.9702

    # Any value,
    transition_matrix = 0.6
    obs_mtx_noise = 0.25 # any value
    prior_mean = np.zeros(shape=state_dim)
    observation_matrix = 1
    prior_cov = np.diag(state_mtx_noise ** 2 / np.sqrt(1. - transition_matrix ** 2) * np.ones(shape=[state_dim, ]))

    model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                          observation_size=observed_dim,
                                          latent_size=state_dim,
                                          initial_state_mean=prior_mean,
                                          initial_state_cov=prior_cov,
                                          state_noise_std=state_mtx_noise,
                                          obs_noise_std=obs_mtx_noise,
                                          transition_matrix=transition_matrix,
                                          observation_matrix=observation_matrix,
                                          mu=mu,
                                          rho=rho,
                                          nonlinear_type="stochastic_volatility")

    return model_obj