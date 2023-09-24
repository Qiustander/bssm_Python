import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.distributions import mvn_tril

tfd = tfp.distributions


def marginal_likelihood(ssm_model,
                        observations,
                        latent_states,
                        final_step_only=True):
    """Calculate the marginal likelihood p(y_{1:t}) given a set of state variables x_{1:t}
    This is specifically for linear Gaussian model.

    Args:
    ssm_model: state space model object
    observations: a (structure of) `Tensor`s, each of shape
        `concat([[num_timesteps, b1, ..., bN], [event_size]])` with scalar
        `event_size` and optional batch dimensions `b1, ..., bN`.
    num_particles: number of ensembles at each step. Could be time-varying
    num_particles: number of ensembles at each step. Could be time-varying
    dampling: Floating-point `Tensor` representing how much to damp the
            update by. Used to mitigate filter divergence. Default value: 1.

    Returns:
        """

    log_marginal_likelihood = forward_pass(
        transition_dist=ssm_model.transition_dist,
        observation_dist=ssm_model.observation_dist,
        transition_fn=ssm_model.transition_fn,
        observation_fn=ssm_model.observation_fn,
        observations=observations,
        filtered_states=latent_states)
    # add likelihood at first time step
    state_prior = ssm_model.transition_dist(0, ssm_model.initial_state_prior.mean())
    first_likelihood = state_prior.log_prob(latent_states[0])

    observation_dist = ssm_model.observation_dist(0, latent_states[0])
    first_likelihood += observation_dist.log_prob(observations[0])

    log_marginal_likelihood = tf.concat([first_likelihood[tf.newaxis, ...], log_marginal_likelihood], axis=0)

    if final_step_only:
        log_marginal_likelihood = tf.reduce_sum(log_marginal_likelihood)

    return log_marginal_likelihood


def forward_pass(transition_fn,
                 observation_fn,
                 transition_dist,
                 observation_dist,
                 observations,
                 filtered_states):
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
    update_step_fn = build_step(
        transition_fn,
        observation_fn,
        transition_dist,
        observation_dist)

    observations_shape = prefer_static.shape(
        tf.nest.flatten(observations)[0])
    dummy_zeros = tf.zeros(observations_shape[1:-1])
    # filtered states 0 -> T-1, next filtered states 1 -> T, observation 1 -> T
    next_filtered_states = tf.roll(filtered_states, shift=-1, axis=0)
    observations = tf.roll(observations, shift=-1, axis=0)

    log_marginal_likelihood, time_step = tf.scan(update_step_fn, elems=(filtered_states[:-1],
                                                                        next_filtered_states[:-1],
                                                                        observations[:-1]),
                                                 initializer=(dummy_zeros,
                                                              tf.cast(dummy_zeros, dtype=tf.int32)))


    return log_marginal_likelihood


def build_step(transition_fn,
               observation_fn,
               transition_dist,
               observation_dist):
    """Build a callable that perform one step for backward smoothing.

    Args:

    Returns:
    backward_pass_step: a callable that updates a BackwardPassState
      from timestep `t` to `t-1`.
    """

    def forward_pass_step(state,
                          observations_states):
        """Run a single step of backward smoothing."""

        log_marginal_likelihood, \
            time_step = _marignal_likelihood_one_step(state, observations_states,
                                                      transition_fn=transition_fn,
                                                      observation_fn=observation_fn,
                                                      transition_dist=transition_dist,
                                                      observation_dist=observation_dist,
                                                      )

        return log_marginal_likelihood, time_step

    return forward_pass_step


def _marignal_likelihood_one_step(state,
                                  observations_states,
                                  transition_fn,
                                  observation_fn,
                                  transition_dist,
                                  observation_dist):
    """A single step of the Kalman Filter.

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
    # If observations are scalar, we can avoid some matrix ops.
    current_state = observations_states[0]
    next_state = observations_states[1]
    observation = observations_states[-1]
    time_step = state[-1]

    state_prior = transition_dist(time_step+1, current_state)
    log_marginal_likelihood = state_prior.log_prob(next_state)

    observation_dist = observation_dist(time_step+1, next_state)
    log_marginal_likelihood += observation_dist.log_prob(observation)

    return (log_marginal_likelihood,
            time_step + 1)
