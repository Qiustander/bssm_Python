import types
from Synthetic_Data.linear_gaussian import gen_data
from Models.ssm_nlg import NonlinearSSM
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from Inference.Kalman.kalman_filter import kalman_filter

tfd = tfp.distributions


class TestKalmanFilter:

    def test_kffilter_univaraite_constant_dynamic_lg(self):
        num_timesteps = 50
        state_dim = 1
        observation_dim = 1
        testcase = 'univariate'
        state_mtx_noise = 1e-11
        obs_mtx_noise = 1e-11
        transition_matrix = 1.
        observation_matrix = 1.

        """
        Generate data 
        """
        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim,
                             state_mtx_noise=state_mtx_noise, obs_mtx_noise=obs_mtx_noise,
                             transition_matrix=transition_matrix, observation_matrix=observation_matrix)

        true_state, observations = ssm_model.simulate(len_time_step=1.1*num_timesteps)
        true_state = true_state[-num_timesteps:]
        observations = observations[-num_timesteps:]

        @tf.function
        def run_kf():
            return kalman_filter(ssm_model=ssm_model,
                                 observations=observations)

        infer_result = run_kf()

        tf.debugging.assert_shapes([(infer_result[0], (num_timesteps, state_dim)), # filtered_means
                                    (infer_result[1], (num_timesteps, state_dim, state_dim)), # filtered_covs
                                    (infer_result[2], (num_timesteps, state_dim)), # predicted_means
                                    (infer_result[3], (num_timesteps, state_dim, state_dim)),# predicted_covs
                                        ])

        # constant state, must converge to this point
        tf.debugging.assert_near(infer_result[0], true_state, atol=1e-6)
        tf.debugging.assert_near(infer_result[2], true_state, atol=1e-6)
        # covariance would not change
        diff_operation = tf.experimental.numpy.diff(infer_result[1], n=1, axis=0)
        tf.debugging.assert_near(diff_operation, tf.zeros_like(diff_operation), atol=1e-6)

        diff_operation = tf.experimental.numpy.diff(infer_result[3], n=1, axis=0)
        tf.debugging.assert_near(diff_operation, tf.zeros_like(diff_operation), atol=1e-6)

        tf.debugging.assert_shapes([(infer_result[0], (num_timesteps, state_dim)), # filtered_means
                                    (infer_result[1], (num_timesteps, state_dim, state_dim)), # filtered_covs
                                    (infer_result[2], (num_timesteps, state_dim)), # predicted_means
                                    (infer_result[3], (num_timesteps, state_dim, state_dim)),# predicted_covs
                                        ])

    def test_kffilter_multivaraite_constant_dynamic_lg(self):
        num_timesteps = 50
        state_dim = 4
        observation_dim = 4
        testcase = 'multivariate'

        """
        Generate data 
        """
        transition_matrix = np.diag(np.random.uniform(low=0.99, high=0.99, size=[state_dim, ]))
        state_mtx_noise = np.diag(np.random.uniform(low=1e-10, high=1e-10, size=[state_dim, ]))
        obs_mtx_noise = np.diag(np.random.uniform(low=1e-10, high=1e-10, size=[observation_dim, ]))
        observation_matrix = np.ones(shape=[observation_dim, state_dim])

        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim,
                             state_mtx_noise=state_mtx_noise, obs_mtx_noise=obs_mtx_noise,
                             transition_matrix=transition_matrix, observation_matrix=observation_matrix)
        true_state, observations = ssm_model.simulate()

        @tf.function
        def run_kf():
            return kalman_filter(ssm_model=ssm_model,
                                 observations=observations)

        infer_result = run_kf()

        # constant state, must converge to this point
        tf.debugging.assert_near(infer_result[0], true_state, atol=1e-6)
        tf.debugging.assert_near(infer_result[2], true_state, atol=1e-6)
        # covariance would not change
        diff_operation = tf.experimental.numpy.diff(infer_result[1], n=1, axis=0)
        tf.debugging.assert_near(diff_operation, tf.zeros_like(diff_operation), atol=1e-6)

        diff_operation = tf.experimental.numpy.diff(infer_result[3], n=1, axis=0)
        tf.debugging.assert_near(diff_operation, tf.zeros_like(diff_operation), atol=1e-6)

        tf.debugging.assert_shapes([(infer_result[0], (num_timesteps, state_dim)), # filtered_means
                                    (infer_result[1], (num_timesteps, state_dim, state_dim)), # filtered_covs
                                    (infer_result[2], (num_timesteps, state_dim)), # predicted_means
                                    (infer_result[3], (num_timesteps, state_dim, state_dim)),# predicted_covs
                                        ])

    def test_kffilter_bad_init_estimate(self):
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
            return kalman_filter(ssm_model=ssm_model,
                                 observations=observations)

        infer_result = run_kf()

        # compare difference between predicted_means and filtered_means
        tf.debugging.assert_near(infer_result[0], true_state, atol=1e-1)

    def test_kffilter_exact_measurement(self):
        # small observation noise would make KF rely on exact observation
        num_timesteps = 50
        state_dim = 1
        observation_dim = 1
        testcase = 'univariate'
        state_mtx_noise = 0.5
        obs_mtx_noise = 1e-6
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

        infer_result = run_kf()

        # compare difference between predicted_means and filtered_means
        tf.debugging.assert_near(infer_result[0], true_state, atol=1e-4)
        tf.debugging.assert_near(infer_result[1], tf.zeros_like(infer_result[1]), atol=1e-6)


    def test_kffilter_noisy_measurement(self):
        # large observation noise would make KF do not update
        num_timesteps = 50
        state_dim = 1
        observation_dim = 1
        testcase = 'univariate'
        state_mtx_noise = 0.5
        obs_mtx_noise = 1e5
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

        ssm_model._initial_state_prior = tfd.MultivariateNormalTriL(
            loc=[100.],
            scale_tril=[[1.]])

        @tf.function
        def run_kf():
            return kalman_filter(ssm_model=ssm_model,
                                 observations=observations)

        infer_result = run_kf()

        # covariance matrix does not change
        diff = tf.experimental.numpy.diff(infer_result[1][10:], n=1, axis=0)
        tf.debugging.assert_near(diff, tf.zeros_like(diff), atol=1e-5)

        diff = tf.experimental.numpy.diff(infer_result[3][10:], n=1, axis=0)
        tf.debugging.assert_near(diff, tf.zeros_like(diff), atol=1e-5)