import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps
from collections import namedtuple
from Utils.smc_utils.smc_utils import ess_below_threshold
from Utils.smc_utils import resampling

weighted_particles = namedtuple(
    'WeightedParticles', ['particles', 'log_weights'])

tfd = tfp.distributions


def bootstrap_particle_filter(ssm_model,
                              observations,
                              num_particles,
                              num_transitions_per_observation=1,
                              resample_criterion=ess_below_threshold,
                              resample_fn='systematic',
                              seed=None):
    """Bootstrap Kalman Filter to observed data.

        """
    if hasattr(ssm_model, 'initial_state_proposal'):
        ssm_model.initial_state_proposal = None

    prior_particles = _particle_filter_initial_weighted_particles(observations,
                                                ssm_model.observation_dist,
                                                ssm_model.initial_state_prior,
                                                ssm_model.initial_state_proposal,
                                                num_particles,
                                                seed=seed)

    filtered_particles = forward_filter_pass(
        transition_fn=ssm_model.observation_dist,
        observation_fn=ssm_model.transition_dist,
        observations=observations,
        initial_particles=prior_particles)

    return filtered_particles


def forward_filter_pass(transition_fn,
                        observation_fn,
                        observations,
                        initial_particles):
    """Run the forward pass in ensembles Kalman filter.

    Args:
      observations:
      scaling_parameters:
      transition_fn: a Python `callable` that accepts (batched) vectors of length
        `state_size`, and returns a `tfd.Distribution` instance, typically a
        `MultivariateNormal`, representing the state transition and covariance.
      observation_fn: a Python `callable` that accepts a (batched) vector of
        length `state_size` and returns a `tfd.Distribution` instance, typically
        a `MultivariateNormal` representing the observation model and covariance.

    Returns:
        filtered_particles
    """
    update_step_fn = build_forward_filter_step(
        transition_fn,
        observation_fn
        )

    filtered_particles = tf.scan(update_step_fn, elems=observations,
                                         initializer=(initial_particles))

    return filtered_particles


def build_forward_filter_step(transition_fn,
                              observation_fn
                              ):
    """Build a callable that perform one step for backward smoothing.

    Args:

    Returns:
    backward_pass_step: a callable that updates a BackwardPassState
      from timestep `t` to `t-1`.
    """

    def forward_pass_step(filtered_particles,
                          observations):
        """Run a single step of backward smoothing."""

        filtered_particles = _bootstrap_one_step(filtered_particles, observations,
                                                                     transition_fn=transition_fn,
                                                                     observation_fn=observation_fn)

        return filtered_particles

    return forward_pass_step


def _bootstrap_one_step(
        filtered_particles, observation, transition_fn, observation_fn):
    """A single step of the EnKF.

    Args:
    state: A `Tensor` of shape
      `concat([[num_timesteps, b1, ..., bN], [state_size]])` with scalar
      `event_size` and optional batch dimensions `b1, ..., bN`.
    observation: A `Tensor` of shape
      `concat([[num_timesteps, b1, ..., bN], [event_size]])` with scalar
      `event_size` and optional batch dimensions `b1, ..., bN`.
    transition_fn: a Python `callable` that accepts (batched) vectors of length
      `state_size`, and returns a `tfd.Distribution` instance, typically a
      `MultivariateNormal`, representing the state transition and covariance.
    observation_fn: a Python `callable` that accepts a (batched) vector of
      length `state_size` and returns a `tfd.Distribution` instance, typically
      a `MultivariateNormal` representing the observation model and covariance.
    Returns:
    updated_state: filtered ensembles
    """



    return new_particles



def _particle_filter_initial_weighted_particles(observations,
                                                observation_fn,
                                                initial_state_prior,
                                                initial_state_proposal,
                                                num_particles,
                                                seed=None):
    """Initialize a set of weighted particles including the first observation."""
    # Propose an initial state.
    if initial_state_proposal is None:
        initial_state = initial_state_prior.sample(num_particles, seed=seed)
        initial_log_weights = ps.zeros_like(
            initial_state_prior.log_prob(initial_state))
    else:
        initial_state = initial_state_proposal.sample(num_particles, seed=seed)
        initial_log_weights = (initial_state_prior.log_prob(initial_state) -
                               initial_state_proposal.log_prob(initial_state))
        # Normalize the initial weights. If we used a proposal, the weights are
        # normalized in expectation, but actually normalizing them reduces variance.
        initial_log_weights = tf.nn.log_softmax(initial_log_weights, axis=0)

    # Return particles weighted by the initial observation.
    return weighted_particles(particles=initial_state, log_weights=initial_log_weights +
                                                                   _compute_observation_log_weights(
                                                                                      step=0,
                                                                                      particles=initial_state,
                                                                                      observations=observations,
                                                                                      observation_fn=observation_fn))


def _compute_observation_log_weights(step,
                                     particles,
                                     observations,
                                     observation_fn,
                                     num_transitions_per_observation=1):
    """Computes particle importance weights from an observation step.

    Args:
    step: int `Tensor` current step.
    particles: Nested structure of `Tensor`s, each of shape
      `concat([[num_particles, b1, ..., bN], event_shape])`, where
      `b1, ..., bN` are optional batch dimensions and `event_shape` may
      differ across `Tensor`s.
    observations: Nested structure of `Tensor`s, each of shape
      `concat([[num_observations, b1, ..., bN], event_shape])`
      where `b1, ..., bN` are optional batch dimensions and `event_shape` may
      differ across `Tensor`s.
    observation_fn: callable with signature
      `observation_dist = observation_fn(step, particles)`, producing
      a batch of distributions over the `observation` at the given `step`,
      one for each particle.
    num_transitions_per_observation: optional int `Tensor` number of times
      to apply the transition model between successive observation steps.
      Default value: `1`.
    Returns:
    log_weights: `Tensor` of shape `concat([num_particles, b1, ..., bN])`.
    """
    with tf.name_scope('compute_observation_log_weights'):

        step_has_observation = (
            # The second of these conditions subsumes the first, but both are
            # useful because the first can often be evaluated statically.
            ps.equal(num_transitions_per_observation, 1) |
            ps.equal(step % num_transitions_per_observation, 0))
        observation_idx = step // num_transitions_per_observation
        observation = tf.nest.map_structure(
            lambda x, step=step: tf.gather(x, observation_idx), observations)

        log_weights = observation_fn(step, particles).log_prob(observation)
        return tf.where(step_has_observation,
                        log_weights,
                        tf.zeros_like(log_weights))
