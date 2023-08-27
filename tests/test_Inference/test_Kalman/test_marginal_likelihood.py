import types
from Synthetic_Data.linear_gaussian import gen_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from Inference.Kalman.kalman_filter import kalman_filter
from Inference.Kalman.marginal_likelihood import marginal_likelihood

"""
Test marginal likelihood function
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

        @tf.function
        def _comp_kf():
            infer_result = kalman_filter(ssm_model=ssm_model,
                                         observations=observations)
            likelihood = marginal_likelihood(ssm_model=ssm_model,
                                             observations=observations,
                                             latent_states=infer_result[0],
                                             final_step_only=False)
            return infer_result, likelihood

        infer_results, computed_likelihood = _comp_kf()

        # compare loglik
        tf.debugging.assert_near(infer_results[-1], computed_likelihood, atol=1e-2)

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

        infer_result = kalman_filter(ssm_model=ssm_model,
                                     observations=observations)

        # me, = plt.plot(infer_result[0], color='black', linewidth=1)
        # tfp, = plt.plot(infer_result_tfp.filtered_means, color='green', linewidth=1)
        # true, = plt.plot(true_state, '-.', color='red', linewidth=1)
        # plt.legend(handles=[tfp, me, true], labels=['tfp', 'me', 'true'])
        # plt.show()

        # compare loglik
        tf.debugging.assert_near(infer_result[-1], infer_result_tfp.log_likelihoods, atol=1e-2)
