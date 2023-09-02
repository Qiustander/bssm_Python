import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from Models.check_argument import *
from Models.ssm_nlg import NonlinearSSM
from Utils.smc_utils.smc_utils import posterior_mean_var
from tensorflow_probability.python.internal import prefer_static as ps
# from Inference.SMC.particle_filter import particle_filter
from Inference.SMC.auxiliary_particle_filter import auxiliary_particle_filter
from Inference.SMC.infer_trajectories import infer_trajectories
from tensorflow_probability.python.experimental.mcmc.particle_filter import infer_trajectories as infer_rej
from tensorflow_probability.python.experimental.mcmc.particle_filter import particle_filter
import pytest
import matplotlib.pyplot as plt
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.distributions import bernoulli
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import linear_gaussian_ssm as lgssm
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import poisson
from tensorflow_probability.python.distributions import sample as sample_dist_lib
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math import gradient
from tests.tf_test_fn import assertAllInRange

tfd = tfp.distributions

num_particles = 300
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


class TestParticleFilter:

    @pytest.mark.parametrize(("num_particles", "seed"),
                             [(num_particles, seed)])
    def test_random_walk(self, num_particles, seed):
        initial_state_prior = jdn.JointDistributionNamed(
            {'position': deterministic.Deterministic(0.)})

        # Biased random walk.
        def particle_dynamics(_, previous_state):
            state_shape = ps.shape(previous_state['position'])
            return jdn.JointDistributionNamed({
                'position':
                    transformed_distribution.TransformedDistribution(
                        bernoulli.Bernoulli(
                            probs=tf.fill(state_shape, 0.75), dtype=tf.float32),
                        shift.Shift(previous_state['position']))
            })

        # Completely uninformative observations allowing a test
        # of the pure dynamics.
        def particle_observations(_, state):
            state_shape = ps.shape(state['position'])
            return uniform.Uniform(
                low=tf.fill(state_shape, -100.), high=tf.fill(state_shape, 100.))

        model_obj._transition_dist = particle_dynamics
        model_obj._observation_dist = particle_observations
        model_obj._initial_state_prior = initial_state_prior

        observations = tf.zeros((9,), dtype=tf.float32)

        trajectories, _ = infer_trajectories(model_obj,
                                             observations=observations,
                                             num_particles=16384,
                                             particle_filter_name='bsf',
                                             resample_ess=0.5,
                                             resample_fn='systematic',
                                             seed=seed)
        position = trajectories['position']

        # The trajectories have the following properties:
        # 1. they lie completely in the range [0, 8]
        assertAllInRange(position, 0., 8.)
        # 2. each step lies in the range [0, 1]
        assertAllInRange(position[1:] - position[:-1], 0., 1.)
        # 3. the expectation and variance of the final positions are 6 and 1.5.
        tf.debugging.assert_near(tf.reduce_mean(position[-1]), 6., atol=0.1)
        tf.debugging.assert_near(tf.math.reduce_variance(position[-1]), 1.5, atol=0.1)

    @pytest.mark.parametrize(("num_particles", "seed"),
                             [(num_particles, seed)])
    def test_data_driven_proposal(self, num_particles, seed):
        num_particles = 100
        observations = tf.convert_to_tensor([60., -179.2, 1337.42])

        # Define a system constrained primarily by observations, where proposing
        # from the dynamics would be a bad fit.
        initial_state_prior = normal.Normal(loc=0., scale=1e6)
        transition_fn = (
            lambda _, previous_state: normal.Normal(loc=previous_state, scale=1e6))
        observation_fn = lambda _, state: normal.Normal(loc=state, scale=0.1)
        initial_state_proposal = normal.Normal(loc=observations[0], scale=0.1)
        proposal_fn = (
            lambda step, state: normal.Normal(  # pylint: disable=g-long-lambda
                loc=tf.ones_like(state) * observations[step + 1],
                scale=1.0))

        model_obj._transition_dist = transition_fn
        model_obj._observation_dist = observation_fn
        model_obj._initial_state_prior = initial_state_prior
        model_obj.proposal_dist = proposal_fn

        """Test Particle_filter
        """

        @tf.function
        def run_tfp_filter():
            (particles,
             log_weights,
             parent_indices,
             likelihood) = particle_filter(observations,
                                           initial_state_prior=model_obj.initial_state_prior,
                                           transition_fn=model_obj.transition_dist,
                                           observation_fn=model_obj.observation_dist,
                                           num_particles=num_particles,
                                           initial_state_proposal=initial_state_proposal,
                                           proposal_fn=proposal_fn,
                                           seed=seed)
            filtered_mean, predicted_mean, \
                filtered_variance, predicted_variance = posterior_mean_var(particles,
                                                                           log_weights,
                                                                           tf.get_static_value(tf.shape(observations))[
                                                                               0])

            return filtered_mean, predicted_mean, \
                filtered_variance, predicted_variance, particles, parent_indices, likelihood

        @tf.function
        def run_pf_filter():
            infer_result = auxiliary_particle_filter(model_obj,
                                                     observations=observations,
                                                     num_particles=num_particles,
                                                     initial_state_proposal=initial_state_proposal,
                                                     resample_ess=0.5,
                                                     resample_fn='systematic',
                                                     seed=seed,
                                                     is_gudied=True)

            return infer_result

        infer_result_tfp = run_tfp_filter()
        infer_result_me = run_pf_filter()

        tf.debugging.assert_near(infer_result_tfp[0], infer_result_me.filtered_mean.numpy(), atol=1e-5)

        """Run trajectory"""

        trajectories, _ = infer_trajectories(model_obj,
                                             observations=observations,
                                             num_particles=num_particles,
                                             particle_filter_name='apf',
                                             initial_state_proposal=initial_state_proposal,
                                             resample_ess=0.5,
                                             resample_fn='systematic',
                                             seed=seed,
                                             is_guided=True)

        trajectories_tfp, _ = infer_rej(
            observations=observations,
            initial_state_prior=initial_state_prior,
            transition_fn=transition_fn,
            observation_fn=observation_fn,
            num_particles=num_particles,
            initial_state_proposal=initial_state_proposal,
            proposal_fn=proposal_fn,
            seed=seed,)

        tf.debugging.assert_near(trajectories, tf.convert_to_tensor(
            tf.convert_to_tensor(
                observations)[..., tf.newaxis] *
            tf.ones([num_particles])), atol=1.0)


def debug_plot(tfp_result, tfp_result2, true_state):
    plt.plot(tfp_result, color='blue', linewidth=1)
    plt.plot(tfp_result2, color='green', linewidth=1)
    plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.show()
