import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.math import linalg
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import normal

tfd = tfp.distributions


def kalman_filter(ssm_model,
                  observations,
                  final_step_only=False,
                  log_likelihood=False):
    """Applies Kalman Filter to observed data.

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
      filtered_mean: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [state_size]])`. The mean of the
        filtered state estimate.
      filtered_cov: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [state_size, state_size]])`.
         The covariance of the filtered state estimate.
      predicted_mean: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [state_size]])`. The prior
        predicted means of the state.
      predicted_cov: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [state_size, state_size]])`
        The prior predicted covariances of the state estimate.
      observation_mean: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [event_size]])`. The prior
        predicted mean of observations.
      observation_cov: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [event_size, event_size]])`. The
        prior predicted covariance of observations.
        """

    observations_shape = prefer_static.shape(
        tf.nest.flatten(observations)[0])
    dummy_zeros = tf.zeros(observations_shape[1:-1])
    initial_state = ssm_model.initial_state_prior.mean() + dummy_zeros[..., tf.newaxis]
    initial_covariance = (
            ssm_model.initial_state_prior.covariance() +
            dummy_zeros[..., tf.newaxis, tf.newaxis])

    (filtered_means, filtered_covs,
     predicted_means, predicted_covs, log_marginal_likelihood) = forward_filter_pass(
        transition_dist=ssm_model.transition_dist,
        observation_dist=ssm_model.observation_dist,
        transition_fn=ssm_model.transition_fn,
        observation_fn=ssm_model.observation_fn,
        observations=observations,
        filtered_means=initial_state, filtered_covs=initial_covariance,
        predicted_means=initial_state, predicted_covs=initial_covariance)

    if final_step_only:
        (filtered_means, filtered_covs,
         predicted_means, predicted_covs, log_marginal_likelihood) = filtered_means[-1, ...], filtered_covs[-1, ...], \
            predicted_means[-1, ...], predicted_covs[-1, ...], log_marginal_likelihood[-1, ...]

    if log_likelihood:
        return tf.reduce_sum(log_marginal_likelihood)

    return (filtered_means, filtered_covs,
            predicted_means, predicted_covs, log_marginal_likelihood)


def forward_filter_pass(transition_fn,
                        observation_fn,
                        transition_dist,
                        observation_dist,
                        observations,
                        filtered_means, filtered_covs,
                        predicted_means, predicted_covs):
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
        observation_fn,
        transition_dist,
        observation_dist)

    observations_shape = prefer_static.shape(
        tf.nest.flatten(observations)[0])
    dummy_zeros = tf.zeros(observations_shape[1:-1])

    (filtered_means, filtered_covs,
     predicted_means, predicted_covs,
     log_marginal_likelihood, time_step) = tf.scan(update_step_fn,
                                                   elems=observations,
                                                   initializer=(filtered_means,
                                                                filtered_covs,
                                                                predicted_means,
                                                                predicted_covs,
                                                                dummy_zeros,
                                                                tf.cast(dummy_zeros, dtype=tf.int32)))
    # one-step predicted mean and covariance
    state_prior = transition_dist(time_step[-1]-1, filtered_means[-1, ...])
    last_mean = state_prior.mean()
    current_transition = transition_fn(time_step[-1]-1,
                                       tf.eye(prefer_static.shape(filtered_means)[-1]))
    last_cov = (tf.matmul(
        current_transition,
        tf.matmul(filtered_covs[-1, ...], current_transition, transpose_b=True)) +
                state_prior.covariance())
    predicted_means = tf.concat([predicted_means[1:], last_mean[tf.newaxis, ...]], axis=0)
    predicted_covs = tf.concat([predicted_covs[1:], last_cov[tf.newaxis, ...]], axis=0)

    return (filtered_means, filtered_covs,
            predicted_means, predicted_covs, log_marginal_likelihood)


def build_forward_filter_step(transition_fn,
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
                          observations):
        """Run a single step of backward smoothing."""

        (filtered_mean,
         filtered_cov,
         predicted_mean,
         predicted_cov,
         log_marginal_likelihood,
         time_step) = _kalman_filter_one_step(state, observations,
                                              transition_fn=transition_fn,
                                              observation_fn=observation_fn,
                                              transition_dist=transition_dist,
                                              observation_dist=observation_dist,
                                              )

        return (filtered_mean,
                filtered_cov,
                predicted_mean,
                predicted_cov,
                log_marginal_likelihood,
                time_step)

    return forward_pass_step


def _kalman_filter_one_step(state,
                            observation,
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
    observation_size_is_static_and_scalar = (observation.shape[-1] == 1)

    current_state = state[0]
    current_covariance = state[1]
    time_step = state[-1]
    current_transition_mtx = transition_fn(time_step,
                                           tf.eye(prefer_static.shape(current_state)[-1]))
    state_prior = transition_dist(time_step, current_state)

    predicted_cov = (tf.matmul(
        current_transition_mtx,
        tf.matmul(current_covariance, current_transition_mtx, transpose_b=True)) +
                     state_prior.covariance())
    predicted_mean = state_prior.mean()

    observation_dist = observation_dist(time_step, predicted_mean)
    observation_mean = observation_dist.mean()
    observation_cov = observation_dist.covariance()

    predicted_obs_mtx = observation_fn(time_step,
                                       tf.eye(prefer_static.shape(current_state)[-1]))
    tmp_obs_cov = tf.matmul(predicted_obs_mtx, predicted_cov)
    residual_covariance = tf.matmul(
        predicted_obs_mtx, tmp_obs_cov, transpose_b=True) + observation_cov

    if observation_size_is_static_and_scalar:
        gain_transpose = tmp_obs_cov / residual_covariance
        # A plain Normal would have event shape `[]`; wrapping with Independent
        # ensures `event_shape=[1]` as required.
        predictive_dist = independent.Independent(
            normal.Normal(loc=observation_mean,
                          scale=tf.sqrt(residual_covariance[..., 0])),
            reinterpreted_batch_ndims=1)
    else:
        chol_residual_cov = tf.linalg.cholesky(residual_covariance)
        gain_transpose = linalg.hpsd_solve(
            residual_covariance, tmp_obs_cov, cholesky_matrix=chol_residual_cov)
        predictive_dist = mvn_tril.MultivariateNormalTriL(
            loc=observation_mean,
            scale_tril=chol_residual_cov)

    filtered_mean = predicted_mean + tf.linalg.matvec(
        gain_transpose,
        (observation - observation_mean),
        transpose_a=True)

    tmp_term = -tf.matmul(predicted_obs_mtx, gain_transpose, transpose_a=True)
    tmp_term = tf.linalg.set_diag(tmp_term, tf.linalg.diag_part(tmp_term) + 1.)
    filtered_cov = (
            tf.matmul(
                tmp_term, tf.matmul(predicted_cov, tmp_term), transpose_a=True) +
            tf.matmul(gain_transpose,
                      tf.matmul(observation_cov, gain_transpose), transpose_a=True))

    log_marginal_likelihood = predictive_dist.log_prob(observation)

    # check positiveness of covariance matrix
    eig_val = tf.linalg.eigvalsh(filtered_cov)
    tf.debugging.assert_greater(eig_val,
                                tf.zeros_like(eig_val),
                                message=f'filtered covariance not positive definite at time step {time_step}')
    eig_val = tf.linalg.eigvalsh(predicted_cov)
    tf.debugging.assert_greater(eig_val,
                                tf.zeros_like(eig_val),
                                message=f'predicted covariance not positive definite at time step {time_step}')
    eig_val = tf.linalg.eigvalsh(residual_covariance)
    tf.debugging.assert_greater(eig_val,
                                tf.zeros_like(residual_covariance),
                                message=f'residual covariance not positive definite at time step {time_step}')

    return (filtered_mean,
            filtered_cov,
            predicted_mean,
            predicted_cov,
            log_marginal_likelihood,
            time_step + 1)
