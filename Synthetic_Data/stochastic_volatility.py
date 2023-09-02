import tensorflow as tf
import numpy as np
from Models.ssm_nlg import NonlinearSSM
"""
Univariate and Multivariate Stochastic Volativity model
"""

def gen_data(testcase='multivariate',
             num_timesteps=200, state_dim=1,
             observed_dim=1):

    if testcase == 'univariate':
        # Univariate
        state_dim = 1
        observed_dim = 1
        state_mtx_noise = 0.9
        obs_mtx_noise = 0.25
        transition_matrix = 0.6
        observation_matrix = 1
        prior_mean = np.zeros(shape=state_dim)
        prior_cov = np.diag(state_mtx_noise**2/np.sqrt(1. - transition_matrix**2)*np.ones(shape=[state_dim, ]))

    elif testcase == 'multivariate':
        # Multivariate
        np.random.seed(seed=123)
        transition_matrix = tf.linalg.diag(np.random.uniform(low=-0.8, high=0.9, size=[state_dim,]))

        state_mtx_noise = np.diag(np.random.uniform(low=0.5, high=1., size=[state_dim, ]))
        obs_mtx_noise = np.diag(np.random.uniform(low=0.5, high=1., size=[observed_dim, ]))
        observation_matrix = 0.5*np.ones(shape=[observed_dim, state_dim])

        noise_cov = tf.matmul(state_mtx_noise, state_mtx_noise, transpose_b=True)
        prior_mean = np.zeros(shape=state_dim)
        operator_coeff = tf.linalg.LinearOperatorFullMatrix(transition_matrix)
        prior_cov = tf.linalg.matvec(tf.linalg.inv(tf.eye(state_dim ** 2, dtype=noise_cov.dtype) -
                                                       tf.linalg.LinearOperatorKronecker(
                                                           [operator_coeff, operator_coeff]).to_dense()),
                                         tf.reshape(noise_cov, [-1]))
        prior_cov = tf.reshape(prior_cov, [state_dim, state_dim]).numpy()

    else:
        raise AttributeError("Wrong input")
    model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                          observation_size=observed_dim,
                                          latent_size=state_dim,
                                          initial_state_mean=prior_mean,
                                          initial_state_cov=prior_cov,
                                          state_noise_std=state_mtx_noise,
                                          obs_noise_std=obs_mtx_noise,
                                          transition_matrix=transition_matrix,
                                          observation_matrix=observation_matrix,
                                          nonlinear_type="linear_gaussian")

    return model_obj
