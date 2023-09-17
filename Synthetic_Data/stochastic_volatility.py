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


def gen_ssm(state_mtx_noise,
            rho,
            mu,
            num_timesteps=200,
            state_dim=1,
            observed_dim=1,
            default_mode='univariate'):

    if default_mode == 'univariate':
        state_mtx_noise = state_mtx_noise
        mu = mu
        rho = rho

        # Any value,
        transition_matrix = 0.6
        obs_mtx_noise = 0.25  # any value
        prior_mean = np.ones(shape=state_dim) * mu
        observation_matrix = 1
        prior_cov = np.diag((state_mtx_noise ** 2 / (1. - rho ** 2)) * np.ones(shape=[state_dim, ]))
    else:

        mu = mu
        rho = rho
        state_mtx_noise = state_mtx_noise

        # Any value,
        transition_matrix = 0.6
        obs_mtx_noise = 0.25  # any value
        prior_mean = mu
        observation_matrix = 1

        noise_cov = tf.matmul(state_mtx_noise, state_mtx_noise, transpose_b=True)

        # stationary mean
        operator_coeff = tf.linalg.LinearOperatorFullMatrix(rho)
        true_cov = tf.linalg.matvec(tf.linalg.inv(tf.eye(state_dim ** 2) -
                                                  tf.linalg.LinearOperatorKronecker(
                                                      [operator_coeff, operator_coeff]).to_dense()),
                                    tf.reshape(noise_cov, [-1]))
        prior_cov = tf.reshape(true_cov, [state_dim, state_dim])

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
                                          nonlinear_type="stochastic_volatility_mv")

    return model_obj
