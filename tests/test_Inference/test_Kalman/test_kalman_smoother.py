import types
from Synthetic_Data.linear_gaussian import gen_data
from Models.ssm_lg import LinearGaussianSSM
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Inference.Kalman.kalman_smoother import kalman_smoother

"""
Test with Kalman Smoother in TFP
"""


class TestKalmanSmoother:

    def test_ksmoother_univaraite_lg(self):
        num_timesteps = 200
        state_dim = 1
        observation_dim = 1
        testcase = 'univariate'
        state_mtx_noise = 0.9
        obs_mtx_noise = 0.25
        transition_matrix = 0.6
        observation_matrix = 1.

        """
        Generate data 
        """
        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim)

        true_state, observations = ssm_model.simulate()

        tfp_model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                       observation_size=1,
                                                       latent_size=1,
                                                       initial_state_mean=np.zeros(shape=state_dim),
                                                       initial_state_cov=np.diag(state_mtx_noise ** 2 / np.sqrt(
                                                           1. - transition_matrix ** 2) * np.ones(shape=[state_dim, ])),
                                                       state_noise_std=state_mtx_noise,
                                                       obs_noise_std=obs_mtx_noise,
                                                       obs_mtx=observation_matrix,
                                                       state_mtx=transition_matrix)
        infer_result_tfp = tfp_model_obj.posterior_marginals(tf.convert_to_tensor(observations,
                                                                             dtype=tfp_model_obj.dtype))

        @tf.function
        def run_kf():
            return kalman_smoother(ssm_model=ssm_model,
                                     observations=observations)

        infer_result = run_kf()

        # compare smoothed_means
        tf.debugging.assert_near(infer_result_tfp[0][5:], infer_result[0][5:], atol=1e-6)
        # compare smoothed_covs
        tf.debugging.assert_near(infer_result_tfp[1][5:], infer_result[1][5:], atol=1e-6)

    def test_kfsmoother_multivaraite_lg(self):
        num_timesteps = 200
        state_dim = 5
        observation_dim = 3
        testcase = 'multivariate'

        """
        Generate data 
        """
        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim)
        np.random.seed(seed=123)

        transition_matrix = tf.linalg.diag(np.random.uniform(low=-0.8, high=0.9, size=[state_dim, ]))
        state_mtx_noise = np.diag(np.random.uniform(low=0.5, high=1., size=[state_dim, ]))
        obs_mtx_noise = np.diag(np.random.uniform(low=0.5, high=1., size=[observation_dim, ]))
        observation_matrix = 0.5 * np.ones(shape=[observation_dim, state_dim])
        noise_cov = tf.matmul(state_mtx_noise, state_mtx_noise, transpose_b=True)
        prior_mean = np.zeros(shape=state_dim)
        operator_coeff = tf.linalg.LinearOperatorFullMatrix(transition_matrix)
        prior_cov = tf.linalg.matvec(tf.linalg.inv(tf.eye(state_dim ** 2, dtype=noise_cov.dtype) -
                                                   tf.linalg.LinearOperatorKronecker(
                                                       [operator_coeff, operator_coeff]).to_dense()),
                                     tf.reshape(noise_cov, [-1]))
        prior_cov = tf.reshape(prior_cov, [state_dim, state_dim]).numpy()

        true_state, observations = ssm_model.simulate()

        tfp_model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                       observation_size=observation_dim,
                                                       latent_size=state_dim,
                                                       initial_state_mean=prior_mean,
                                                       initial_state_cov=prior_cov,
                                                       state_noise_std=state_mtx_noise,
                                                       obs_noise_std=obs_mtx_noise,
                                                       obs_mtx=observation_matrix,
                                                       state_mtx=transition_matrix)
        infer_result_tfp = tfp_model_obj.posterior_marginals(tf.convert_to_tensor(observations,
                                                                             dtype=tfp_model_obj.dtype))

        @tf.function
        def run_kf():
            return kalman_smoother(ssm_model=ssm_model,
                                     observations=observations)

        infer_result = run_kf()

        # compare smoothed_means
        tf.debugging.assert_near(infer_result_tfp[0], infer_result[0], atol=1e-6)
        # compare smoothed_covs
        tf.debugging.assert_near(infer_result_tfp[1], infer_result[1], atol=1e-6)
