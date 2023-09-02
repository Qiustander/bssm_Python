import types
from Synthetic_Data.linear_gaussian import gen_data
from Models.ssm_lg import LinearGaussianSSM
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Inference.Kalman.kalman_filter import kalman_filter
from Inference.Kalman.kalman_smoother import kalman_smoother
from Models.approximate_model import approximate_model, initial_mode_by_ekf, lg_latent_update

"""
Test with Kalman Filter in TFP
"""


class TestApproxModel:

    def test_initial_ekf_approx_univaraite(self):
        # from lg to lg
        num_timesteps = 100
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

        @tf.function
        def run_kf(model):
            return kalman_filter(ssm_model=model,
                                 observations=observations)

        infer_result = run_kf(ssm_model)
        init_model = initial_mode_by_ekf(ssm_model, observations=observations)

        tf.debugging.assert_near(init_model.transition_fn(0, tf.constant([1.])),
                                 ssm_model.transition_fn(0, tf.constant([1.])), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_fn(0, tf.constant([1.])),
                                 ssm_model.observation_fn(0, tf.constant([1.])), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_dist(0, tf.constant([1.])).mean(),
                                 ssm_model.observation_dist(0, tf.constant([1.])).mean(), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_dist(0, tf.constant([1.])).covariance(),
                                 ssm_model.observation_dist(0, tf.constant([1.])).covariance(), atol=1e-10)
        tf.debugging.assert_near(init_model.transition_dist(0, tf.constant([1.])).mean(),
                                 ssm_model.transition_dist(0, tf.constant([1.])).mean(), atol=1e-10)
        tf.debugging.assert_near(init_model.transition_dist(0, tf.constant([1.])).covariance(),
                                 ssm_model.transition_dist(0, tf.constant([1.])).covariance(), atol=1e-10)

        tf.debugging.assert_near(init_model.initial_state_prior.mean(), ssm_model.initial_state_prior.mean(),
                                 atol=1e-10)
        tf.debugging.assert_near(init_model.initial_state_prior.covariance(),
                                 ssm_model.initial_state_prior.covariance(), atol=1e-10)

        infer_result_approx = run_kf(init_model)

        # compare loglik
        tf.debugging.assert_near(infer_result[-1][5:], infer_result_approx[-1][5:], atol=1e-6)
        # compare filtered_means
        tf.debugging.assert_near(infer_result[0][5:], infer_result_approx[0][5:], atol=1e-6)
        # compare filtered_covs
        tf.debugging.assert_near(infer_result[1][5:], infer_result_approx[1][5:], atol=1e-6)
        # compare predicted_means
        tf.debugging.assert_near(infer_result[2][5:], infer_result_approx[2][5:], atol=1e-6)
        # compare predicted_covs
        tf.debugging.assert_near(infer_result[3][5:], infer_result_approx[3][5:], atol=1e-6)

    def test_initial_ekf_approx_multivaraite(self):
        # from lg to lg
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

        @tf.function
        def run_kf(model):
            return kalman_filter(ssm_model=model,
                                 observations=observations)

        infer_result = run_kf(ssm_model)
        init_model = initial_mode_by_ekf(ssm_model, observations=observations)

        tf.debugging.assert_near(init_model.transition_fn(0, tf.ones([state_dim, ])),
                                 ssm_model.transition_fn(0, tf.ones([state_dim, ])), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_fn(0, tf.ones([state_dim, ])),
                                 ssm_model.observation_fn(0, tf.ones([state_dim, ])), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_dist(0, tf.ones([state_dim, ])).mean(),
                                 ssm_model.observation_dist(0, tf.ones([state_dim, ])).mean(), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_dist(0, tf.ones([state_dim, ])).covariance(),
                                 ssm_model.observation_dist(0, tf.ones([state_dim, ])).covariance(), atol=1e-10)
        tf.debugging.assert_near(init_model.transition_dist(0, tf.ones([state_dim, ])).mean(),
                                 ssm_model.transition_dist(0, tf.ones([state_dim, ])).mean(), atol=1e-10)
        tf.debugging.assert_near(init_model.transition_dist(0, tf.ones([state_dim, ])).covariance(),
                                 ssm_model.transition_dist(0, tf.ones([state_dim, ])).covariance(), atol=1e-10)

        tf.debugging.assert_near(init_model.initial_state_prior.mean(), ssm_model.initial_state_prior.mean(),
                                 atol=1e-10)
        tf.debugging.assert_near(init_model.initial_state_prior.covariance(),
                                 ssm_model.initial_state_prior.covariance(), atol=1e-10)

        infer_result_approx = run_kf(init_model)

        # compare loglik
        tf.debugging.assert_near(infer_result[-1][5:], infer_result_approx[-1][5:], atol=1e-6)
        # compare filtered_means
        tf.debugging.assert_near(infer_result[0][5:], infer_result_approx[0][5:], atol=1e-6)
        # compare filtered_covs
        tf.debugging.assert_near(infer_result[1][5:], infer_result_approx[1][5:], atol=1e-6)
        # compare predicted_means
        tf.debugging.assert_near(infer_result[2][5:], infer_result_approx[2][5:], atol=1e-6)
        # compare predicted_covs
        tf.debugging.assert_near(infer_result[3][5:], infer_result_approx[3][5:], atol=1e-6)

    def test_updatelg_univaraite(self):
        num_timesteps = 50
        state_dim = 1
        observation_dim = 1
        testcase = 'univariate'

        """
        Generate data 
        """
        ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                             state_dim=state_dim, observed_dim=observation_dim)

        true_state, observations = ssm_model.simulate()

        # @tf.function
        # def run_kf(model):
        #     return kalman_filter(ssm_model=model,
        #                          observations=observations)
        #
        # infer_result = run_kf(ssm_model)
        init_model = initial_mode_by_ekf(ssm_model, observations=observations)
        init_lg_latent_estimate, _, _ = kalman_smoother(init_model, observations)
        init_model = lg_latent_update(ssm_model=ssm_model,
                                      observations=observations,
                                      lg_latent_estimate=init_lg_latent_estimate)
        tf.debugging.assert_near(init_model.transition_fn(0, tf.constant([1.])),
                                 ssm_model.transition_fn(0, tf.constant([1.])), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_fn(0, tf.constant([1.])),
                                 ssm_model.observation_fn(0, tf.constant([1.])), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_dist(0, tf.constant([1.])).mean(),
                                 ssm_model.observation_dist(0, tf.constant([1.])).mean(), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_dist(0, tf.constant([1.])).covariance(),
                                 ssm_model.observation_dist(0, tf.constant([1.])).covariance(), atol=1e-10)
        tf.debugging.assert_near(init_model.transition_dist(0, tf.constant([1.])).mean(),
                                 ssm_model.transition_dist(0, tf.constant([1.])).mean(), atol=1e-10)
        tf.debugging.assert_near(init_model.transition_dist(0, tf.constant([1.])).covariance(),
                                 ssm_model.transition_dist(0, tf.constant([1.])).covariance(), atol=1e-10)
        tf.debugging.assert_near(init_model.initial_state_prior.mean(), ssm_model.initial_state_prior.mean(),
                                 atol=1e-10)
        tf.debugging.assert_near(init_model.initial_state_prior.covariance(),
                                 ssm_model.initial_state_prior.covariance(), atol=1e-10)

    def test_updatelg_multivaraite(self):
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

        init_model = initial_mode_by_ekf(ssm_model, observations=observations)
        init_lg_latent_estimate, _, _ = kalman_smoother(init_model, observations)
        init_model = lg_latent_update(ssm_model=ssm_model,
                                      observations=observations,
                                      lg_latent_estimate=init_lg_latent_estimate)

        tf.debugging.assert_near(init_model.transition_fn(0, tf.ones([state_dim, ])),
                                 ssm_model.transition_fn(0, tf.ones([state_dim, ])), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_fn(0, tf.ones([state_dim, ])),
                                 ssm_model.observation_fn(0, tf.ones([state_dim, ])), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_dist(0, tf.ones([state_dim, ])).mean(),
                                 ssm_model.observation_dist(0, tf.ones([state_dim, ])).mean(), atol=1e-10)
        tf.debugging.assert_near(init_model.observation_dist(0, tf.ones([state_dim, ])).covariance(),
                                 ssm_model.observation_dist(0, tf.ones([state_dim, ])).covariance(), atol=1e-10)
        tf.debugging.assert_near(init_model.transition_dist(0, tf.ones([state_dim, ])).mean(),
                                 ssm_model.transition_dist(0, tf.ones([state_dim, ])).mean(), atol=1e-10)
        tf.debugging.assert_near(init_model.transition_dist(0, tf.ones([state_dim, ])).covariance(),
                                 ssm_model.transition_dist(0, tf.ones([state_dim, ])).covariance(), atol=1e-10)

        tf.debugging.assert_near(init_model.initial_state_prior.mean(), ssm_model.initial_state_prior.mean(),
                                 atol=1e-10)
        tf.debugging.assert_near(init_model.initial_state_prior.covariance(),
                                 ssm_model.initial_state_prior.covariance(), atol=1e-10)

    def test_approx_univaraite(self):
        # from lg to lg
        num_timesteps = 100
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

        @tf.function
        def run_kf(model):
            return kalman_filter(ssm_model=model,
                                 observations=observations)

        infer_result = run_kf(ssm_model)

        # @tf.function
        def approx_model(model):
            return approximate_model(model, observations=observations)

        approx_ssm, approx_result = approx_model(ssm_model)
        infer_result_approx = run_kf(approx_ssm)

        tf.debugging.assert_near(approx_ssm.transition_fn(0, tf.constant([1.])),
                                 ssm_model.transition_fn(0, tf.constant([1.])), atol=1e-10)
        tf.debugging.assert_near(approx_ssm.observation_fn(0, tf.constant([1.])),
                                 ssm_model.observation_fn(0, tf.constant([1.])), atol=1e-10)
        tf.debugging.assert_near(approx_ssm.observation_dist(0, tf.constant([1.])).mean(),
                                 ssm_model.observation_dist(0, tf.constant([1.])).mean(), atol=1e-10)
        tf.debugging.assert_near(approx_ssm.observation_dist(0, tf.constant([1.])).covariance(),
                                 ssm_model.observation_dist(0, tf.constant([1.])).covariance(), atol=1e-10)
        tf.debugging.assert_near(approx_ssm.transition_dist(0, tf.constant([1.])).mean(),
                                 ssm_model.transition_dist(0, tf.constant([1.])).mean(), atol=1e-10)
        tf.debugging.assert_near(approx_ssm.transition_dist(0, tf.constant([1.])).covariance(),
                                 ssm_model.transition_dist(0, tf.constant([1.])).covariance(), atol=1e-10)

        tf.debugging.assert_near(approx_ssm.initial_state_prior.mean(), ssm_model.initial_state_prior.mean(),
                                 atol=1e-10)
        tf.debugging.assert_near(approx_ssm.initial_state_prior.covariance(),
                                 ssm_model.initial_state_prior.covariance(), atol=1e-10)

        # compare loglik
        tf.debugging.assert_near(infer_result[-1][5:], infer_result_approx[-1][5:], atol=1e-6)
        # compare filtered_means
        tf.debugging.assert_near(infer_result[0][5:], infer_result_approx[0][5:], atol=1e-6)
        # compare filtered_covs
        tf.debugging.assert_near(infer_result[1][5:], infer_result_approx[1][5:], atol=1e-6)
        # compare predicted_means
        tf.debugging.assert_near(infer_result[2][5:], infer_result_approx[2][5:], atol=1e-6)
        # compare predicted_covs
        tf.debugging.assert_near(infer_result[3][5:], infer_result_approx[3][5:], atol=1e-6)



def debug_plot(first_result, second_result, true_state):
    plt.plot(first_result, color='blue', linewidth=1)
    plt.plot(second_result, color='green', linewidth=1)
    plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.show()
