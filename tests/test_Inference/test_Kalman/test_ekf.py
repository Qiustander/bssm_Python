import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from Models.ssm_nlg import NonlinearSSM
from Inference.Kalman.extended_kalman_filter import extended_kalman_filter
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
        num_timesteps = obs_len = 200
        observation_size = 1
        state_dim = 1

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observation_size,
                                              latent_size=state_dim,
                                              initial_state_mean=0.,
                                              initial_state_cov=1.,
                                              state_noise_std=1e-11,
                                              obs_noise_std=1e-11,
                                              nonlinear_type="constant_dynamic_univariate_test")
        true_state, observation = model_obj.simulate()

        @tf.function
        def run_method():
            infer_result = extended_kalman_filter(model_obj, observation)
            return infer_result

        infer_result = run_method()

        # constant observation, must converge to this point
        tf.debugging.assert_near(infer_result[0][50:], true_state[50:], atol=1e-6)
        tf.debugging.assert_near(infer_result[2][51:], true_state[50:], atol=1e-6)
        # covariance would not change
        diff_operation = infer_result[1][1:][30:] - infer_result[1][:-1][30:]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-6)

        diff_operation = infer_result[3][1:][30:] - infer_result[3][:-1][30:]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-6)

        tf.debugging.assert_shapes([(infer_result[0], (obs_len, state_dim)), # filtered_means
                                    (infer_result[1], (obs_len, state_dim, state_dim)), # filtered_covs
                                    (infer_result[2], (obs_len+1, state_dim)), # predicted_means
                                    (infer_result[3], (obs_len+1, state_dim, state_dim)),# predicted_covs
                                        ])

    def test_multivariate_model_shape(self):
        num_timesteps = obs_len = 200
        state_dim = 4
        observation_size = 3
        # observation = np.ones([obs_len, 3])

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observation_size,
                                              latent_size=state_dim,
                                              initial_state_mean=np.array(tf.random.normal([4,])),
                                              initial_state_cov=np.diag([0.01, 0.01, 0.01, 0.01]),
                                              state_noise_std=np.diag([1e-11]*4),
                                              obs_noise_std=np.diag([1e-1]*3),
                                              nonlinear_type="constant_dynamic_multivariate_test")
        true_state, observation = model_obj.simulate()

        @tf.function
        def run_method():
            infer_result = extended_kalman_filter(model_obj, observation)
            return infer_result

        infer_result = run_method()
        # constant observation, must converge to this point
        tf.debugging.assert_near(infer_result[0][50:, 0], true_state[50:, 0], atol=1e-6)
        tf.debugging.assert_near(infer_result[0][50:, 1], true_state[50:, 1], atol=1e-6)
        tf.debugging.assert_near(infer_result[0][50:, 2], true_state[50:, 2], atol=1e-1)
        tf.debugging.assert_near(infer_result[0][50:, 3], true_state[50:, 3], atol=1e-1)

        tf.debugging.assert_near(infer_result[2][51:, 0], true_state[50:, 0], atol=1e-6)
        tf.debugging.assert_near(infer_result[2][51:, 1], true_state[50:, 1], atol=1e-6)
        tf.debugging.assert_near(infer_result[2][51:, 2], true_state[50:, 2], atol=1e-0)
        tf.debugging.assert_near(infer_result[2][51:, 3], true_state[50:, 3], atol=1e-0)

        # covariance would not change
        diff_operation = infer_result[1][1:][30:] - infer_result[1][:-1][30:]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-2)

        diff_operation = infer_result[3][1:][30:] - infer_result[3][:-1][30:]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-2)

        tf.debugging.assert_shapes([(infer_result[0], (obs_len, state_dim)), # filtered_means
                                    (infer_result[1], (obs_len, state_dim, state_dim)), # filtered_covs
                                    (infer_result[2], (obs_len+1, state_dim)), # predicted_means
                                    (infer_result[3], (obs_len+1, state_dim, state_dim)),# predicted_covs
                                        ])

def debug_plot(tfp_result, true_state):
    plt.plot(tfp_result, color='blue', linewidth=1)
    plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.show()
