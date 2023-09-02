import types
from Synthetic_Data.linear_gaussian import gen_data
from Models.ssm_lg import LinearGaussianSSM
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Inference.Kalman.kalman_filter import kalman_filter
from Inference.Kalman.marginal_likelihood import marginal_likelihood

"""
Test with Kalman Filter in TFP
"""


class TestMarginalLikelihood:

    def test_marginal_likelihood_univaraite_lg(self):
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

        @tf.function
        def run_kf():
            return kalman_filter(ssm_model=ssm_model,
                                     observations=observations)

        infer_result = run_kf()
        filter_state = infer_result[0]
        filter_likelihood = infer_result[-1]

        mar_likelihood = marginal_likelihood(ssm_model=ssm_model, observations=observations,
                                             latent_states=filter_state, final_step_only=False)
        tf.debugging.assert_near(mar_likelihood, filter_likelihood, atol=1e-6)

    def test_marginal_likelihood_multivaraite_lg(self):
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
        def run_kf():
            return kalman_filter(ssm_model=ssm_model,
                                     observations=observations)

        infer_result = run_kf()
        filter_state = infer_result[0]
        filter_likelihood = infer_result[-1]

        mar_likelihood = marginal_likelihood(ssm_model=ssm_model, observations=observations,
                                             latent_states=filter_state, final_step_only=False)
        tf.debugging.assert_near(mar_likelihood, filter_likelihood, atol=1e-6)
