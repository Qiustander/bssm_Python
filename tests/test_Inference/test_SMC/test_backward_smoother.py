import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from Models.check_argument import *
from Models.ssm_nlg import NonlinearSSM
from Utils.smc_utils.smc_utils import posterior_mean_var
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.experimental.mcmc.particle_filter import infer_trajectories, particle_filter, \
    reconstruct_trajectories
from Inference.SMC.forward_filter_backward_smoother import forward_filter_backward_smoother
from Inference.SMC.forward_filter_backward_sampling import forward_filter_backward_sampling
import pytest
import matplotlib.pyplot as plt
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

tfd = tfp.distributions


# TODO: rewrite the backward smoother

class TestTrajectoryReconstruct:
    num_particles = 200
    seed = 457
    num_timesteps = 100
    model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                          observation_size=1,
                                          latent_size=1,
                                          initial_state_mean=0,
                                          initial_state_cov=1.,
                                          state_noise_std=0.1,
                                          obs_noise_std=0.2,
                                          nonlinear_type="nlg_sin_exp")

    @pytest.mark.parametrize(("num_particles", "seed", "num_timesteps", "model_obj"),
                             [(num_particles, seed, num_timesteps, model_obj)])
    def test_reconstruct_trajectory(self, num_particles, seed, num_timesteps, model_obj):
        true_state, observations = model_obj.simulate(seed)

        # TFP BSF filter
        @tf.function
        def run_tfp_smoother():
            (particles,
             log_weights,
             parent_indices,
             likelihood) = particle_filter(observations,
                                           initial_state_prior=model_obj.initial_state_prior,
                                           transition_fn=model_obj.transition_dist,
                                           observation_fn=model_obj.observation_dist,
                                           num_particles=num_particles,
                                           seed=seed)
            filtered_mean, predicted_mean, \
                filtered_variance, predicted_variance = posterior_mean_var(particles,
                                                                           log_weights,
                                                                           tf.get_static_value(tf.shape(observations))[
                                                                               0])
            weighted_trajectories = reconstruct_trajectories(particles, parent_indices)
            from tensorflow_probability.python.experimental.mcmc import weighted_resampling

            # Resample all steps of the trajectories using the final weights.
            resample_indices = weighted_resampling.resample_systematic(log_probs=log_weights[-1],
                                                                       event_size=num_particles,
                                                                       sample_shape=(),
                                                                       seed=seed)

            trajectories = tf.nest.map_structure(
                lambda x: mcmc_util.index_remapping_gather(x,  # pylint: disable=g-long-lambda
                                                           resample_indices,
                                                           axis=1),
                weighted_trajectories)

            return trajectories, filtered_mean, predicted_mean, \
                filtered_variance, predicted_variance, particles, parent_indices, likelihood, resample_indices

        @tf.function
        def run_ffbs():
            infer_result_smoother = forward_filter_backward_smoother(model_obj,
                                                                     observations,
                                                                     particle_filter_name='bsf',
                                                                     resample_ess=0.5,
                                                                     resample_fn='systematic',
                                                                     num_particles=num_particles,
                                                                     seed=seed)
            # infer_result_smoother = forward_filter_backward_sampling(model_obj,
            #                                                          observations,
            #                                                          particle_filter_name='bsf',
            #                                                          resample_ess=0.5,
            #                                                          resample_fn='stratified',
            #                                                          num_particles=num_particles,
            #                                                          is_one_trajectory=False)

            return infer_result_smoother

        infer_result_tfp = run_tfp_smoother()
        infer_result_me = run_ffbs()
        debug_plot(infer_result_tfp[0].numpy().mean(axis=1), infer_result_me.smoother_mean, true_state)


def debug_plot(tfp_result, tfp_result2, true_state):
    plt.plot(tfp_result, color='blue', linewidth=1)
    plt.plot(tfp_result2, color='green', linewidth=1)
    plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.show()
