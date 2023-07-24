import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from Models.ssm_nlg import NonlinearSSM
from Inference.SMC.extend_kalman_particle_filter import extended_kalman_particle_filter
from Models.check_argument import *
import os.path as pth
import os
import matplotlib.pyplot as plt
tfd = tfp.distributions

"""
Check consistence
https://cs.adelaide.edu.au/~ianr/Teaching/Estimation/LectureNotes2.pdf

1. load fixed state and bssm result for comparison. (same as numerical test)
2. compare shape
3. constant dynamic
"""

class TestExtendKalmanFilter:

    def test_univariate_model(self):
        obs_len = 200
        state_dim = 1
        observation = np.ones(obs_len, )
        size_y, observation = check_y(observation.astype("float32"))  # return time length and feature numbers
        num_timesteps, observation_size = size_y

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observation_size,
                                              latent_size=state_dim,
                                              initial_state_mean=0.,
                                              initial_state_cov=1.,
                                              state_noise_std=0.1,
                                              obs_noise_std=0.1,
                                              nonlinear_type="constant_dynamic_univariate_test")

        infer_result = extended_kalman_particle_filter(model_obj,
                                                 observation,
                                                 resample_ess=0.5,
                                                 num_particles=200)

        # constant observation, must converge to this point
        tf.debugging.assert_near(infer_result.filtered_mean, observation, atol=1e-1)
        tf.debugging.assert_near(infer_result.predicted_mean[1:], observation[1:], atol=1e-1)
        # covariance would not change
        diff_operation = infer_result.filtered_variance[-obs_len//4+1:] - infer_result.filtered_variance[1:obs_len//4]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-1)

        diff_operation = infer_result.predicted_variance[-obs_len//4+1:] - infer_result.predicted_variance[1:obs_len//4]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-1)

        tf.debugging.assert_shapes([(infer_result.filtered_mean, (obs_len, state_dim)), # filtered_means
                                    (infer_result.filtered_variance, (obs_len, state_dim, state_dim)), # filtered_covs
                                    (infer_result.predicted_mean, (obs_len, state_dim)), # predicted_means
                                    (infer_result.predicted_variance, (obs_len, state_dim, state_dim)),# predicted_covs
                                        ])

    def test_multivariate_model_shape(self):
        obs_len = 600
        state_dim = 4
        # observation = np.ones([obs_len, 3])
        observation = np.stack([np.ones([obs_len,]), 2*np.ones([obs_len,]), 4*np.ones([obs_len,])], axis=-1)
        size_y, observation = check_y(observation.astype("float32"))  # return time length and feature numbers
        num_timesteps, observation_size = size_y

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observation_size,
                                              latent_size=state_dim,
                                              initial_state_mean=np.array([0., 0., 0., 0.]),
                                              initial_state_cov=np.diag([1, 2, 1, 1.5]),
                                              state_noise_std=np.diag([0.1, 0.1, 0.5, 0.1]),
                                              obs_noise_std=np.diag([0.1, 0.1, 0.1]),
                                              nonlinear_type="constant_dynamic_multivariate_test")

        infer_result = extended_kalman_particle_filter(model_obj,
                                                 observation,
                                                 resample_ess=0.5,
                                                 num_particles=1000)

        # constant observation, must converge to this point
        tf.debugging.assert_near(infer_result.filtered_mean[50:, 0], observation[50:, 0], atol=1e-1)
        tf.debugging.assert_near(infer_result.filtered_mean[50:, 1], observation[50:, 1], atol=1e-1)
        tf.debugging.assert_near(infer_result.filtered_mean[50:, 2] + 0.1*infer_result.filtered_mean[50:, 3], observation[50:, 2], atol=1e-1)

        tf.debugging.assert_near(infer_result.predicted_mean[50:, 0], observation[50:, 0], atol=1e-1)
        tf.debugging.assert_near(infer_result.predicted_mean[50:, 1], observation[50:, 1], atol=1e-1)
        tf.debugging.assert_near(infer_result.predicted_mean[50:, 2] + 0.1*infer_result.predicted_mean[50:, 3], observation[50:, 2], atol=1e-1)

        # covariance would not change
        diff_operation = infer_result.filtered_variance[-obs_len//4:] - infer_result.filtered_variance[obs_len//4:obs_len//2]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-2)

        diff_operation = infer_result.predicted_variance[1:][30:] - infer_result.predicted_variance[:-1][30:]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-2)

        tf.debugging.assert_shapes([(infer_result.filtered_mean, (obs_len, state_dim)), # filtered_means
                                    (infer_result.filtered_variance, (obs_len, state_dim, state_dim)), # filtered_covs
                                    (infer_result.predicted_mean, (obs_len, state_dim)), # predicted_means
                                    (infer_result.predicted_variance, (obs_len, state_dim, state_dim)),# predicted_covs
                                        ])


def debug_plot(tfp_result, true_state):
    plt.plot(tfp_result, color='blue', linewidth=1)
    plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.show()
