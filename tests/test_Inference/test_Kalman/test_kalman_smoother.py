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


class TestKalmanFilter:

    def test_kffilter_univaraite_lg(self):
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

        infer_result = kalman_smoother(ssm_model=ssm_model,
                                     observations=observations)

        # me, = plt.plot(infer_result[0], color='black', linewidth=1)
        # tfp, = plt.plot(infer_result_tfp[0], color='green', linewidth=1)
        # true, = plt.plot(true_state, '-.', color='red', linewidth=1)
        # plt.legend(handles=[tfp, me, true], labels=['tfp', 'me', 'true'])
        # plt.show()

        # compare smoothed_means
        tf.debugging.assert_near(infer_result_tfp[0][5:], infer_result[0][5:], atol=1e-6)
        # compare smoothed_covs
        tf.debugging.assert_near(infer_result_tfp[1][5:], infer_result[1][5:], atol=1e-6)

# TODO:finish the tst
    def test_kffilter_multivaraite_lg(self):
        num_timesteps = 200
        state_dim = 5
        observation_dim = 3
        testcase = 'multivariate'

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
        infer_result_tfp = tfp_model_obj.forward_filter(tf.convert_to_tensor(observations,
                                                                             dtype=tfp_model_obj.dtype))

        infer_result = kalman_filter(ssm_model=ssm_model,
                                     observations=observations)

        # me, = plt.plot(infer_result[0], color='black', linewidth=1)
        # tfp, = plt.plot(infer_result_tfp.filtered_means, color='green', linewidth=1)
        # true, = plt.plot(true_state, '-.', color='red', linewidth=1)
        # plt.legend(handles=[tfp, me, true], labels=['tfp', 'me', 'true'])
        # plt.show()

        # compare loglik
        tf.debugging.assert_near(infer_result[-1], infer_result_tfp.log_likelihoods, atol=1e-2)
        # compare filtered_means
        tf.debugging.assert_near(infer_result[0], infer_result_tfp.filtered_means, atol=1e-1)
        # compare filtered_covs
        tf.debugging.assert_near(infer_result[1], infer_result_tfp.filtered_covs, atol=1e-1)
        # compare predicted_means
        tf.debugging.assert_near(infer_result[2][1:, ...], infer_result_tfp.predicted_means, atol=1e-1)
        # compare predicted_covs
        tf.debugging.assert_near(infer_result[3][1:, ...], infer_result_tfp.predicted_covs, atol=1e-1)