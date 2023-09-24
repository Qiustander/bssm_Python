import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.math import linalg
from collections import namedtuple

tfd = tfp.distributions

ekf_results = namedtuple(
    'ExtendedKalmanFilterResults', ['filtered_means', 'filtered_covs',
                                    'predicted_means', 'predicted_covs', 'log_marginal_likelihood'])


def extended_kalman_filter(ssm_model, observations, iterative_num=0):
    """Applies an Extended Kalman Filter to observed data.

    The [Extended Kalman Filter](
    https://en.wikipedia.org/wiki/Extended_Kalman_filter) is a nonlinear version
    of the Kalman filter, in which the transition function is linearized by
    first-order Taylor expansion around the current mean and covariance of the
    state estimate.

    Args:
      ssm_model: state space model object
      observations: a (structure of) `Tensor`s, each of shape
        `concat([[num_timesteps, b1, ..., bN], [event_size]])` with scalar
        `event_size` and optional batch dimensions `b1, ..., bN`.
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
      log_marginal_likelihood: a (structure of) `Tensor`(s) of shape
        `[num_timesteps, b1, ..., bN]`. Log likelihood of the observations with
        respect to the observation.
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
        transition_fn_grad=ssm_model.transition_fn_grad,
        transition_fn=ssm_model.transition_dist,
        observation_fn=ssm_model.observation_dist,
        observation_fn_grad=ssm_model.observation_fn_grad,
        observations=observations,
        filtered_means=initial_state, filtered_covs=initial_covariance,
        predicted_means=initial_state, predicted_covs=initial_covariance,
        iterative_num=iterative_num)

    return ekf_results(filtered_means=filtered_means, filtered_covs=filtered_covs,
                       predicted_means=predicted_means,
                       predicted_covs=predicted_covs,
                       log_marginal_likelihood=log_marginal_likelihood)


def forward_filter_pass(transition_fn_grad,
                        transition_fn,
                        observation_fn,
                        observation_fn_grad,
                        observations,
                        filtered_means,
                        filtered_covs,
                        predicted_means,
                        predicted_covs,
                        iterative_num):
    """Run the forward pass in extended Kalman filter.

    Args:
      filtered_means:
      observations:
      transition_fn: a Python `callable` that accepts (batched) vectors of length
        `state_size`, and returns a `tfd.Distribution` instance, typically a
        `MultivariateNormal`, representing the state transition and covariance.
      observation_fn: a Python `callable` that accepts a (batched) vector of
        length `state_size` and returns a `tfd.Distribution` instance, typically
        a `MultivariateNormal` representing the observation model and covariance.
      transition_fn_grad: a Python `callable` that accepts a (batched) vector
        of length `state_size` and returns a (batched) matrix of shape
        `[state_size, state_size]`, representing the Jacobian of `transition_fn`.
      observation_fn_grad: a Python `callable` that accepts a (batched) vector
        of length `state_size` and returns a (batched) matrix of size
        `[state_size, event_size]`, representing the Jacobian of `observation_fn`.

    Returns:
      filtered_means
      filtered_covs
      predicted_means
      predicted_covs
      log_marginal_likelihood
    """
    update_step_fn = build_forward_filter_step(
        transition_fn_grad,
        transition_fn,
        observation_fn,
        observation_fn_grad,
        iterative_num)

    observations_shape = prefer_static.shape(
        tf.nest.flatten(observations)[0])
    dummy_zeros = tf.zeros(observations_shape[1:-1])

    (filtered_means, filtered_covs,
     predicted_means, predicted_covs, log_marginal_likelihood, time_step) = tf.scan(update_step_fn,
                                                                                    elems=observations,
                                                                                    initializer=(filtered_means,
                                                                                                 filtered_covs,
                                                                                                 predicted_means,
                                                                                                 predicted_covs,
                                                                                                 dummy_zeros,
                                                                                                 dummy_zeros))

    # one-step predicted mean and covariance
    state_prior = transition_fn(time_step[-1], filtered_means[-1, ...])
    last_mean = state_prior.mean()

    current_jacobian = transition_fn_grad(time_step[-1], filtered_means[-1, ...])
    last_cov = (tf.matmul(
        current_jacobian,
        tf.matmul(filtered_covs[-1, ...], current_jacobian, transpose_b=True)) +
                state_prior.covariance())
    predicted_means = tf.concat([predicted_means, last_mean[tf.newaxis, ...]], axis=0)
    predicted_covs = tf.concat([predicted_covs, last_cov[tf.newaxis, ...]], axis=0)

    return (filtered_means, filtered_covs,
            predicted_means, predicted_covs, log_marginal_likelihood)


def build_forward_filter_step(transition_fn_grad,
                              transition_fn,
                              observation_fn,
                              observation_fn_grad,
                              iterative_num):
    """Build a callable that perform one step for backward smoothing.

    Args:
    transition_fn_grad: callable taking the mu_t|t
      as `Tensor` argument, and returning a `LinearOperator`
      of shape `[latent_size, latent_size]`.

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
         time_step) = _extended_kalman_filter_one_step(state=state, observation=observations,
                                                       transition_fn=transition_fn,
                                                       observation_fn=observation_fn,
                                                       transition_jacobian_fn=transition_fn_grad,
                                                       observation_jacobian_fn=observation_fn_grad,
                                                       iterative_num=iterative_num)

        return (filtered_mean,
                filtered_cov,
                predicted_mean,
                predicted_cov,
                log_marginal_likelihood,
                time_step)

    return forward_pass_step


def _extended_kalman_filter_one_step(
        state, observation, iterative_num,
        transition_fn,
        observation_fn,
        transition_jacobian_fn,
        observation_jacobian_fn):
    """A single step of the EKF.

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
    transition_jacobian_fn: a Python `callable` that accepts a (batched) vector
      of length `state_size` and returns a (batched) matrix of shape
      `[state_size, state_size]`, representing the Jacobian of `transition_fn`.
    observation_jacobian_fn: a Python `callable` that accepts a (batched) vector
      of length `state_size` and returns a (batched) matrix of size
      `[state_size, event_size]`, representing the Jacobian of `observation_fn`.
    Returns:
    updated_state:
            filtered_means
            filtered_covs
            predicted_means
            predicted_covs
            log_marginal_likelihood
    """

    # If observations are scalar, we can avoid some matrix ops.
    observation_size_is_static_and_scalar = (observation.shape[-1] == 1)

    current_state = state[0]
    current_covariance = state[1]
    time_step = state[-1]
    current_jacobian = transition_jacobian_fn(time_step, current_state)
    state_prior = transition_fn(time_step, current_state)

    predicted_cov = (tf.matmul(
        current_jacobian,
        tf.matmul(current_covariance, current_jacobian, transpose_b=True)) +
                     state_prior.covariance())
    predicted_mean = state_prior.mean()

    observation_dist = observation_fn(time_step, predicted_mean)
    observation_mean = observation_dist.mean()
    observation_cov = observation_dist.covariance()

    predicted_jacobian = observation_jacobian_fn(time_step, predicted_mean)
    tmp_obs_cov = tf.matmul(predicted_jacobian, predicted_cov)
    residual_covariance = tf.matmul(
        predicted_jacobian, tmp_obs_cov, transpose_b=True) + observation_cov

    if observation_size_is_static_and_scalar:
        gain_transpose = tmp_obs_cov / residual_covariance
    else:
        chol_residual_cov = tf.linalg.cholesky(residual_covariance)
        gain_transpose = linalg.hpsd_solve(
            residual_covariance, tmp_obs_cov, cholesky_matrix=chol_residual_cov)

    filtered_mean = predicted_mean + tf.linalg.matvec(
        gain_transpose,
        (observation - observation_mean),
        transpose_a=True)

    def _loop_body(last_idx, last_diff, last_mean,
                   last_predicted_jacobian, last_gain_transpose, last_residual_covariance, last_correction):
        new_idx = last_idx + 1

        predicted_jacobian_iekf = observation_jacobian_fn(time_step, last_mean)
        tmp_obs_cov_iekf = tf.matmul(predicted_jacobian_iekf, predicted_cov)
        residual_covariance_iekf = tf.matmul(
            predicted_jacobian_iekf, tmp_obs_cov_iekf, transpose_b=True) + \
                                   observation_fn(time_step, last_mean).covariance()

        if observation_size_is_static_and_scalar:
            gain_transpose_iekf = tmp_obs_cov_iekf / residual_covariance_iekf
        else:
            chol_residual_cov_iekf = tf.linalg.cholesky(residual_covariance_iekf)
            gain_transpose_iekf = linalg.hpsd_solve(
                residual_covariance_iekf, tmp_obs_cov_iekf, cholesky_matrix=chol_residual_cov_iekf)

        correction = observation - observation_fn(time_step, last_mean).mean() - \
                     tf.linalg.matvec(predicted_jacobian_iekf, (predicted_mean - last_mean))
        new_mean = predicted_mean + tf.linalg.matvec(
            gain_transpose_iekf,
            correction,
            transpose_a=True)

        new_diff = tf.reduce_mean((new_mean - last_mean) ** 2)

        return new_idx, new_diff, new_mean, \
            predicted_jacobian_iekf, gain_transpose_iekf, residual_covariance_iekf, observation - correction

    # inner loop for iterative EKF
    init_loop = (tf.constant(0., dtype=predicted_mean.dtype),
                 tf.constant(1., dtype=predicted_mean.dtype),
                 filtered_mean,
                 predicted_jacobian,
                 gain_transpose,
                 residual_covariance,
                 observation_mean)

    def _loop_condition(idx, diff, updated_mean,
                        updated_predicted_jacobian,
                        updated_gain_transpose, updated_residual_covariance, _):
        return tf.logical_and(
            tf.greater(diff, tf.constant(1e-4, dtype=diff.dtype)),
            tf.less(idx, iterative_num))

    _, _, filtered_mean, predicted_jacobian, \
        gain_transpose, residual_covariance, \
        observation_mean = tf.while_loop(cond=_loop_condition,
                                         body=_loop_body,
                                         loop_vars=init_loop,
                                         shape_invariants=(tf.TensorShape([]), tf.TensorShape([]),
                                                           filtered_mean.shape,
                                                           predicted_jacobian.shape,
                                                           gain_transpose.shape,
                                                           residual_covariance.shape,
                                                           observation_mean.shape))

    # calculate likelihood
    # TODO: likelihood not correct for iekf
    observation_cov = observation_fn(time_step, filtered_mean).covariance()
    tmp_term = -tf.matmul(predicted_jacobian, gain_transpose, transpose_a=True)
    tmp_term = tf.linalg.set_diag(tmp_term, tf.linalg.diag_part(tmp_term) + 1.)
    filtered_cov = (
            tf.matmul(
                tmp_term, tf.matmul(predicted_cov, tmp_term), transpose_a=True) +
            tf.matmul(gain_transpose,
                      tf.matmul(observation_cov, gain_transpose), transpose_a=True))

    # if observation_size_is_static_and_scalar:
    #     # A plain Normal would have event shape `[]`; wrapping with Independent
    #     # ensures `event_shape=[1]` as required.
    #     predictive_dist = independent.Independent(
    #         normal.Normal(loc=observation_mean,
    #                       scale=tf.sqrt(residual_covariance[..., 0])),
    #         reinterpreted_batch_ndims=1)
    #
    # else:
    predictive_dist = mvn_tril.MultivariateNormalTriL(
        loc= observation_mean,
        scale_tril=tf.linalg.cholesky(residual_covariance))

    log_marginal_likelihood = predictive_dist.log_prob(observation)

    return (filtered_mean,
            filtered_cov,
            predicted_mean,
            predicted_cov,
            log_marginal_likelihood,
            time_step + 1)
