import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from Models.ssm_nlg import NonlinearSSM
from Inference.Kalman.ensemble_kalman_filter import ensemble_kalman_filter
from Models.check_argument import *
import os.path as pth
import os
import matplotlib.pyplot as plt
tfd = tfp.distributions

""""""

# TODO: compare with kf with linear gaussian

class TestEnsembleKalmanFilter:

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
                                              state_noise_std=1e-11,
                                              obs_noise_std=1e-1,
                                              nonlinear_type="constant_dynamic_univariate_test")
        true_state, observation = model_obj.simulate()

        # @tf.function
        def run_method():
            infer_result = ensemble_kalman_filter(model_obj, observation, num_particles=200)
            return infer_result

        infer_result = run_method()

        # constant observation, must converge to this point
        tf.debugging.assert_near(infer_result[0][50:], true_state[50:], atol=1e-2)
        tf.debugging.assert_near(infer_result[2][51:], true_state[50:], atol=1e-1)
        # covariance would not change
        diff_operation = infer_result[1][1:][30:] - infer_result[1][:-1][30:]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-3)

        diff_operation = infer_result[3][1:][30:] - infer_result[3][:-1][30:]
        tf.debugging.assert_near(diff_operation, tf.zeros(diff_operation.shape), atol=1e-3)

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
        observation = np.stack([np.ones([obs_len,]), 2*np.ones([obs_len,]), 4*np.ones([obs_len,])], axis=-1)
        size_y, observation = check_y(observation.astype("float32"))  # return time length and feature numbers
        num_timesteps, observation_size = size_y

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
            infer_result = ensemble_kalman_filter(model_obj, observation, num_particles=100)
            return infer_result

        infer_result = run_method()

        tf.debugging.assert_shapes([(infer_result[0], (obs_len, state_dim)), # filtered_means
                                    (infer_result[1], (obs_len, state_dim, state_dim)), # filtered_covs
                                    (infer_result[2], (obs_len+1, state_dim)), # predicted_means
                                    (infer_result[3], (obs_len+1, state_dim, state_dim)),# predicted_covs
                                        ])

def debug_plot(tfp_result, true_state):
    plt.plot(tfp_result, color='blue', linewidth=1)
    plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.show()
