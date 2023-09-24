import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from Synthetic_Data.linear_gaussian import gen_data
from Models.ssm_nlg import NonlinearSSM
from Inference.Kalman.extended_kalman_filter import extended_kalman_filter
from Inference.Kalman.kalman_filter import kalman_filter
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

    def test_ekf_univaraite_constant_dynamic(self):
        num_timesteps = obs_len = 100
        observation_size = 1
        state_dim = 1

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observation_size,
                                              latent_size=state_dim,
                                              initial_state_mean=0.,
                                              initial_state_cov=1.,
                                              state_noise_std=1e-8,
                                              obs_noise_std=1e-8,
                                              nonlinear_type="constant_dynamic_univariate_test")
        true_state, observation = model_obj.simulate()

        @tf.function
        def run_method():
            infer_result = extended_kalman_filter(model_obj, observation)
            return infer_result

        infer_result = run_method()

        # constant state, must converge to this point
        tf.debugging.assert_near(infer_result[0], true_state, atol=1e-6)
        tf.debugging.assert_near(infer_result[2][1:], true_state, atol=1e-6)
        # covariance would not change
        diff_operation = tf.experimental.numpy.diff(infer_result[1], n=1, axis=0)
        tf.debugging.assert_near(diff_operation, tf.zeros_like(diff_operation), atol=1e-6)

        diff_operation = tf.experimental.numpy.diff(infer_result[3][1:], n=1, axis=0)
        tf.debugging.assert_near(diff_operation, tf.zeros_like(diff_operation), atol=1e-6)

        tf.debugging.assert_shapes([(infer_result[0], (obs_len, state_dim)),  # filtered_means
                                    (infer_result[1], (obs_len, state_dim, state_dim)),  # filtered_covs
                                    (infer_result[2], (obs_len + 1, state_dim)),  # predicted_means
                                    (infer_result[3], (obs_len + 1, state_dim, state_dim)),  # predicted_covs
                                    ])

    def test_ekf_mnultivariate_constant_dynamic(self):
        num_timesteps = obs_len = 100
        state_dim = 5
        observation_size = 3
        # observation = np.ones([obs_len, 3])

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observation_size,
                                              latent_size=state_dim,
                                              initial_state_mean=np.array(tf.random.normal([4, ])),
                                              initial_state_cov=np.diag([0.01, 0.01, 0.01, 0.01]),
                                              state_noise_std=np.diag([1e-10] * 4),
                                              obs_noise_std=np.diag([1e-10] * 4),
                                              nonlinear_type="constant_dynamic_multivariate_test")

        true_state, observation = model_obj.simulate()

        @tf.function
        def run_method():
            infer_result = extended_kalman_filter(model_obj, observation)
            return infer_result

        infer_result = run_method()
        # constant observation, must converge to this point
        tf.debugging.assert_near(infer_result[0], true_state, atol=1e-6)
        tf.debugging.assert_near(infer_result[2][1:], true_state, atol=1e-6)
        # covariance would not change
        diff_operation = tf.experimental.numpy.diff(infer_result[1], n=1, axis=0)
        tf.debugging.assert_near(diff_operation, tf.zeros_like(diff_operation), atol=1e-6)

        diff_operation = tf.experimental.numpy.diff(infer_result[3][1:], n=1, axis=0)
        tf.debugging.assert_near(diff_operation, tf.zeros_like(diff_operation), atol=1e-6)

        tf.debugging.assert_shapes([(infer_result[0], (obs_len, state_dim)),  # filtered_means
                                    (infer_result[1], (obs_len, state_dim, state_dim)),  # filtered_covs
                                    (infer_result[2], (obs_len + 1, state_dim)),  # predicted_means
                                    (infer_result[3], (obs_len + 1, state_dim, state_dim)),  # predicted_covs
                                    ])

    def test_ekffilter_bad_init_estimate(self):
        # bad initialization would converge to true states
        num_timesteps = 50
        state_dim = 1
        observation_dim = 1
        testcase = 'univariate'
        state_mtx_noise = 0.5
        obs_mtx_noise = 1e-2
        transition_matrix = 0.6
        observation_matrix = 1.

        """
        Generate data 
        """
        prior_cov = np.diag((state_mtx_noise ** 2 / (1. - transition_matrix ** 2)) * np.ones(shape=[state_dim, ]))
        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim,
                             state_mtx_noise=state_mtx_noise, obs_mtx_noise=obs_mtx_noise,
                             transition_matrix=transition_matrix, observation_matrix=observation_matrix,
                             prior_mean=0., prior_cov=prior_cov)

        true_state, observations = ssm_model.simulate(len_time_step=1.1 * num_timesteps)
        true_state = true_state[-num_timesteps:]
        observations = observations[-num_timesteps:]

        # bad init
        ssm_model._initial_state_prior = tfd.MultivariateNormalTriL(
            loc=[100.],
            scale_tril=[[400.]])

        @tf.function
        def run_kf():
            return extended_kalman_filter(ssm_model=ssm_model,
                                          observations=observations)

        infer_result = run_kf()

        # compare difference between predicted_means and filtered_means
        tf.debugging.assert_near(infer_result[0], true_state, atol=1e-1)

    def test_ekffilter_linear_case(self):
        # perform as the KF with LGSSM
        num_timesteps = 50
        state_dim = 1
        observation_dim = 1
        testcase = 'univariate'
        state_mtx_noise = 0.5
        obs_mtx_noise = 1e-2
        transition_matrix = 0.6
        observation_matrix = 1.

        """
        Generate data 
        """
        prior_cov = np.diag((state_mtx_noise ** 2 / (1. - transition_matrix ** 2)) * np.ones(shape=[state_dim, ]))
        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim,
                             state_mtx_noise=state_mtx_noise, obs_mtx_noise=obs_mtx_noise,
                             transition_matrix=transition_matrix, observation_matrix=observation_matrix,
                             prior_mean=0., prior_cov=prior_cov)

        true_state, observations = ssm_model.simulate(len_time_step=1.1 * num_timesteps)
        true_state = true_state[-num_timesteps:]
        observations = observations[-num_timesteps:]

        @tf.function
        def run_kf():
            return kalman_filter(ssm_model=ssm_model,
                                 observations=observations)

        @tf.function
        def run_ekf():
            return extended_kalman_filter(ssm_model=ssm_model,
                                          observations=observations)

        infer_result = run_kf()
        infer_result_ekf = run_ekf()

        # compare difference between ekf and kf
        tf.debugging.assert_near(infer_result[0], infer_result_ekf[0], atol=1e-6)

    def test_iekf_ekffilter(self):
        # iekf log marginal likelihood must larger than ekf
        num_timesteps = 100

        ssm_model = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=1,
                                              latent_size=1,
                                              initial_state_mean=-0.4,
                                              initial_state_cov=1. / np.sqrt(1 - 0.5 ** 2),
                                              mu_state=-0.4,
                                              rho_state=0.5,
                                              state_noise_std=1.,
                                              obs_noise_std=0.5,
                                              nonlinear_type="nlg_ar_exp")

        true_state, observations = ssm_model.simulate(len_time_step=1.1 * num_timesteps)
        true_state = true_state[-num_timesteps:]
        observations = observations[-num_timesteps:]

        @tf.function
        def run_ekf():
            return extended_kalman_filter(ssm_model=ssm_model,
                                          observations=observations)

        @tf.function
        def run_iekf():
            return extended_kalman_filter(ssm_model=ssm_model,
                                          observations=observations, iterative_num=20)

        @tf.function
        def run_iekf2():
            return extended_kalman_filter(ssm_model=ssm_model,
                                          observations=observations, iterative_num=5)

        infer_result_ekf = run_ekf()
        infer_result_iekf = run_iekf()
        infer_result_iekf_10 = run_iekf2()

        tf.debugging.assert_greater(tf.reduce_sum(infer_result_iekf[-1]), tf.reduce_sum(infer_result_ekf[-1]))
        tf.debugging.assert_greater(tf.reduce_sum(infer_result_iekf_10[-1]), tf.reduce_sum(infer_result_ekf[-1]))


def debug_plot(tfp_result, true_state):
    plt.plot(tfp_result, color='blue', linewidth=1)
    plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.show()
