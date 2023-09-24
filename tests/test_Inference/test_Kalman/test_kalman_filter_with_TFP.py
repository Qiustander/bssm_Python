import types
from Synthetic_Data.linear_gaussian import gen_data
from Models.ssm_lg import LinearGaussianSSM
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Inference.Kalman.kalman_filter import kalman_filter

"""
Test with Kalman Filter in TFP
"""


class TestKalmanFilter:

    def test_kffilter_univaraite_lg(self):
        num_timesteps = 200
        state_dim = 1
        observation_dim = 1
        testcase = 'univariate'
        state_mtx_noise = 0.5
        obs_mtx_noise = 0.1
        transition_matrix = 0.6
        observation_matrix = 1.

        """
        Generate data 
        """
        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim,
                             state_mtx_noise=state_mtx_noise, obs_mtx_noise=obs_mtx_noise,
                             transition_matrix=transition_matrix, observation_matrix=observation_matrix)

        true_state, observations = ssm_model.simulate()

        tfp_model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                       observation_size=1,
                                                       latent_size=1,
                                                       initial_state_mean=0.,
                                                       initial_state_cov=np.diag((state_mtx_noise**2/(1. - transition_matrix**2))*np.ones(shape=[state_dim, ])),
                                                       state_noise_std=state_mtx_noise,
                                                       obs_noise_std=obs_mtx_noise,
                                                       obs_mtx=observation_matrix,
                                                       state_mtx=transition_matrix)
        infer_result_tfp = tfp_model_obj.forward_filter(tf.convert_to_tensor(observations,
                                                                             dtype=tfp_model_obj.dtype))

        @tf.function
        def run_kf():
            return kalman_filter(ssm_model=ssm_model,
                                 observations=observations)

        infer_result = run_kf()

        # compare loglik
        tf.debugging.assert_near(infer_result[-1], infer_result_tfp.log_likelihoods, atol=1e-6)
        # compare filtered_means
        tf.debugging.assert_near(infer_result[0], infer_result_tfp.filtered_means, atol=1e-6)
        # compare filtered_covs
        tf.debugging.assert_near(infer_result[1], infer_result_tfp.filtered_covs, atol=1e-6)
        # compare predicted_means
        tf.debugging.assert_near(infer_result[2], infer_result_tfp.predicted_means, atol=1e-6)
        # compare predicted_covs
        tf.debugging.assert_near(infer_result[3], infer_result_tfp.predicted_covs, atol=1e-6)

    def test_kffilter_multivaraite_lg(self):
        num_timesteps = 200
        state_dim = 5
        observation_dim = 3
        testcase = 'multivariate'

        """
        Generate data 
        """
        transition_matrix = np.diag(np.random.uniform(low=-0.6, high=0.6, size=[state_dim, ]))
        state_mtx_noise = np.diag(np.random.uniform(low=0.2, high=0.5, size=[state_dim, ]))
        obs_mtx_noise = np.diag(np.random.uniform(low=0.05, high=0.1, size=[observation_dim, ]))
        observation_matrix = 0.5 * np.ones(shape=[observation_dim, state_dim])

        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim,
                             state_mtx_noise=state_mtx_noise, obs_mtx_noise=obs_mtx_noise,
                             transition_matrix=transition_matrix, observation_matrix=observation_matrix)

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
        infer_result_tfp = tfp_model_obj.forward_filter(tf.convert_to_tensor(observations,
                                                                             dtype=tfp_model_obj.dtype))

        @tf.function
        def run_kf():
            return kalman_filter(ssm_model=ssm_model,
                                 observations=observations)

        infer_result = run_kf()

        # compare loglik
        tf.debugging.assert_near(infer_result[-1], infer_result_tfp.log_likelihoods, atol=1e-6)
        # compare filtered_means
        tf.debugging.assert_near(infer_result[0], infer_result_tfp.filtered_means, atol=1e-6)
        # compare filtered_covs
        tf.debugging.assert_near(infer_result[1], infer_result_tfp.filtered_covs, atol=1e-6)
        # compare predicted_means
        tf.debugging.assert_near(infer_result[2], infer_result_tfp.predicted_means, atol=1e-6)
        # compare predicted_covs
        tf.debugging.assert_near(infer_result[3], infer_result_tfp.predicted_covs, atol=1e-6)
