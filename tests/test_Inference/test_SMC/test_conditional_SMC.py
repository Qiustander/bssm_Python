import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from Inference.SMC.bootstrap_particle_filter import bootstrap_particle_filter

from Synthetic_Data.stochastic_volatility import gen_ssm
import matplotlib.pyplot as plt

tfd = tfp.distributions


class TestConditionalSMC:
    num_particles = 300
    seed = 457
    num_timesteps = 100

    @pytest.mark.parametrize(("num_particles", "seed", "num_timesteps"),
                             [(num_particles, seed, num_timesteps)])
    def test_univariate_model_single_chain(self, num_particles, seed, num_timesteps):
        observation_size = 1
        state_dim = 1

        ssm_model = gen_ssm(num_timesteps=num_timesteps,
                            state_dim=state_dim, observed_dim=observation_size)

        true_state, observations = ssm_model.simulate()

        reference_trajectory = tf.random.normal(true_state.shape)

        @tf.function
        def run_method():
            infer_result = bootstrap_particle_filter(ssm_model,
                                                     observations,
                                                     resample_ess=0.5,
                                                     is_conditional=True,
                                                     conditional_sample=reference_trajectory,
                                                     resample_fn='multinomial',
                                                     num_particles=num_particles)
            return infer_result

        infer_result = run_method()

        # zeroth index must the same
        tf.debugging.assert_equal(infer_result.particles[:, 0], reference_trajectory)
        tf.debugging.assert_equal(infer_result.parent_indices[:, 0],
                                  tf.zeros(tf.shape(infer_result.parent_indices[:, 0]),
                                           dtype=infer_result.parent_indices.dtype))

    @pytest.mark.parametrize(("num_particles", "seed", "num_timesteps"),
                             [(num_particles, seed, num_timesteps)])
    def test_univariate_model_multiple_chain(self, num_particles, seed, num_timesteps):
        observation_size = 1
        state_dim = 1
        N_CHAINS = 5

        ssm_model = gen_ssm(num_timesteps=num_timesteps,
                            state_dim=state_dim, observed_dim=observation_size)

        true_state, observations = ssm_model.simulate()
        replicate_observations = tf.tile(tf.expand_dims(observations, axis=1), multiples=[1, N_CHAINS, 1])

        reference_trajectory = tf.random.normal(replicate_observations.shape)

        # replicate
        sigma = tf.random.uniform([N_CHAINS, 1, 1])+0.2
        mu = tf.random.normal([N_CHAINS, 1])
        rho = tf.random.normal([N_CHAINS, 1])
        ssm_model._initial_state_prior = tfd.MultivariateNormalTriL(
            loc=mu,
            scale_tril=sigma)
        transition_fn = lambda t, x: tf.cast(
            mu * (1. - rho) + rho * x, dtype=sigma.dtype)
        ssm_model._transition_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=transition_fn(t, x),
            scale_tril=sigma)

        @tf.function
        def run_method():
            infer_result = bootstrap_particle_filter(ssm_model,
                                                     replicate_observations,
                                                     resample_ess=0.5,
                                                     is_conditional=True,
                                                     conditional_sample=reference_trajectory,
                                                     resample_fn='multinomial',
                                                     num_particles=num_particles)
            return infer_result

        infer_result = run_method()

        # zeroth index must the same
        tf.debugging.assert_equal(infer_result.particles[:, 0], reference_trajectory)
        tf.debugging.assert_equal(infer_result.parent_indices[:, 0],
                                  tf.zeros(tf.shape(infer_result.parent_indices[:, 0]),
                                           dtype=infer_result.parent_indices.dtype))