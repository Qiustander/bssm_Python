import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.math import linalg
import numpy as np

tfd = tfp.distributions


@tf.function
def unscented_kalman_filter(ssm_model, observations, alpha=1e-2, beta=2., kappa=1.):
    """Applies an Unscented Kalman Filter to observed data.

    Based on Unscented Kalman filter, Särkkä (2013) p.107 (UKF)

    Args:
      ssm_model: state space model object
      observations: a (structure of) `Tensor`s, each of shape
        `concat([[num_timesteps, b1, ..., bN], [event_size]])` with scalar
        `event_size` and optional batch dimensions `b1, ..., bN`.
      alpha: scaling factor
      beta: scaling factor
      kappa: scaling factor
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

    initial_state = ssm_model.initial_state_prior.mean()
    initial_covariance = ssm_model.initial_state_prior.covariance()

    (filtered_means, filtered_covs,
     predicted_means, predicted_covs, log_marginal_likelihood) = forward_filter_pass(
        transition_fn=ssm_model.transition_fn,
        transition_noise=ssm_model.transition_noise_fn.covariance(),
        observation_fn=ssm_model.observation_fn,
        observation_noise=ssm_model.observation_noise_fn.covariance(),
        observations=observations,
        filtered_means=initial_state, filtered_covs=initial_covariance,
        predicted_means=initial_state, predicted_covs=initial_covariance,
        scaling_parameters=(alpha, beta, kappa, ssm_model.state_dim))

    return (filtered_means, filtered_covs,
            predicted_means, predicted_covs, log_marginal_likelihood)


def forward_filter_pass(transition_fn,
                        observation_fn,
                        transition_noise,
                        observation_noise,
                        scaling_parameters,
                        observations,
                        filtered_means, filtered_covs,
                        predicted_means, predicted_covs):
    """Run the forward pass in unscented Kalman filter.

    Args:
      predicted_means:
      observations:
      scaling_parameters:
      transition_fn: a Python `callable` that accepts (batched) vectors of length
        `state_size`, and returns a `tfd.Distribution` instance, typically a
        `MultivariateNormal`, representing the state transition and covariance.
      observation_fn: a Python `callable` that accepts a (batched) vector of
        length `state_size` and returns a `tfd.Distribution` instance, typically
        a `MultivariateNormal` representing the observation model and covariance.

    Returns:
      filtered_means
      filtered_covs
      predicted_means
      predicted_covs
      log_marginal_likelihood
    """
    update_step_fn = build_forward_filter_step(
        transition_fn,
        observation_fn,
        transition_noise,
        observation_noise,
        scaling_parameters)

    observations_shape = prefer_static.shape(
        tf.nest.flatten(observations)[0])
    dummy_zeros = tf.zeros(observations_shape[1:-1])

    (filtered_means, filtered_covs,
     predicted_means, predicted_covs, log_marginal_likelihood) = tf.scan(update_step_fn,
                                                                         elems=observations,
                                                                         initializer=(filtered_means,
                                                                                      filtered_covs,
                                                                                      predicted_means,
                                                                                      predicted_covs,
                                                                                      dummy_zeros))

    return (filtered_means, filtered_covs,
            predicted_means, predicted_covs, log_marginal_likelihood)


def build_forward_filter_step(transition_fn,
                              observation_fn,
                              transition_noise,
                              observation_noise,
                              scaling_parameters):
    """Build a callable that perform one step for forward filtering.

    Args:

    Returns:
    backward_pass_step: a callable that updates a BackwardPassState
      from timestep `t` to `t-1`.
    """
    alpha, beta, kappa, state_dim = scaling_parameters
    lamda = alpha ** 2 * (state_dim.__float__() + kappa) - state_dim.__float__()
    num_sigma = 2 * state_dim.numpy() + 1

    # determinstic sigma points
    sigma_weight_mean = np.ones((num_sigma, 1)) / (2. * (lamda + state_dim.__float__()))
    sigma_weight_mean[0] = lamda / (lamda + state_dim.__float__())
    sigma_weight_cov = sigma_weight_mean.copy()
    sigma_weight_cov[0] = sigma_weight_cov[0] + 1. - alpha ** 2 + beta

    def forward_pass_step(state,
                          observations):
        """Run a single step of forward filtering."""

        (filtered_mean,
         filtered_cov,
         predicted_mean,
         predicted_cov,
         log_marginal_likelihood) = _unscented_kalman_filter_one_step(state, observations,
                                                                     transition_fn=transition_fn,
                                                                     transition_noise=transition_noise,
                                                                     observation_fn=observation_fn,
                                                                     observation_noise=observation_noise,
                                                                     sigma_weight_mean=sigma_weight_mean,
                                                                     sigma_weight_cov=sigma_weight_cov,
                                                                     num_sigma=num_sigma,
                                                                     lamda=lamda)

        return (filtered_mean,
                filtered_cov,
                predicted_mean,
                predicted_cov,
                log_marginal_likelihood)

    return forward_pass_step


def _unscented_kalman_filter_one_step(
        state, observation, transition_fn, observation_fn, transition_noise, observation_noise,
        sigma_weight_mean, sigma_weight_cov, num_sigma, lamda):
    """A single step of the UKF.

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
    current_state, current_cov, predict_state, predict_cov, _ = state
    observation_size_is_static_and_scalar = (observation.shape[-1] == 1)

    chol_cov_mtx = tf.linalg.cholesky(predict_cov)  # lower triangular L
    indices = tf.range(num_sigma)
    sigma_points_x = tf.vectorized_map(
        lambda x: _sigma_samples(predict_state, chol_cov_mtx, tf.sqrt(lamda + tf.cast((num_sigma-1)//2,
                                                                              dtype=predict_state.dtype)), x),
        indices)

    ############### Estimation already mu_t|t-1
    sigma_y = tf.vectorized_map(lambda x:
                                observation_fn(x), sigma_points_x)  # 2M+1 xM
    estimated_y = tf.squeeze(tf.transpose(sigma_y) @ sigma_weight_mean, axis=-1)

    current_variance = _weight_covariance(sigma_y, weights_mean=sigma_weight_mean,
                                               weights_cov=sigma_weight_cov) + observation_noise
    current_covariance = _weight_covariance(sigma_y, sigma_points_x - predict_state,
                                                 weights_mean=sigma_weight_mean,
                                                 weights_cov=sigma_weight_cov,
                                                 need_mean=False)

    ########### Correction mu_t|t
    gamma_t = observation - estimated_y

    if observation_size_is_static_and_scalar:
        kalman_gain = current_covariance / current_variance
        filtered_state = predict_state + tf.squeeze(kalman_gain, axis=-1) * gamma_t
    else:
        kalman_gain = tf.transpose(tf.linalg.solve(
            current_variance, current_covariance, adjoint=True
        ))
        filtered_state = predict_state + kalman_gain._matmul(gamma_t)
    filtered_cov = predict_cov - kalman_gain @ current_variance @ tf.transpose(kalman_gain)

    ############## prediction for next state mu_t+1|t
    chol_next_state = tf.linalg.cholesky(filtered_cov)
    sigma_points_pred = tf.vectorized_map(lambda x:
                                          _sigma_samples(filtered_state, chol_next_state,
                                                              tf.sqrt(lamda + tf.cast((num_sigma-1)//2,
                                                                              dtype=predict_state.dtype)), x), indices)

    sigma_x_pred = tf.vectorized_map(lambda x:
                                     transition_fn(x), tf.stack(sigma_points_pred))
    predict_state = tf.squeeze(tf.transpose(sigma_x_pred) @ sigma_weight_mean, -1)  # s
    predict_cov = _weight_covariance(sigma_x_pred, weights_mean=sigma_weight_mean,
                                          weights_cov=sigma_weight_cov) + transition_noise

    # log_marginal_likelihood = predictive_dist.log_prob(observation)
    # chol_noise = tf.linalg.cholesky(observation_noise)
    # inv_noise = tf.linalg.inv(chol_noise)
    # residual_covariance = tf.matmul(
    #     inv_noise, gamma_t, transpose_a=True)
    #
    # log_marginal_likelihood = -0.5*tf.math.log(2.*np.pi) +2.*tf.math.log(tf.linalg.tensor_diag_part(chol_noise))
    log_marginal_likelihood = 0.

    return (filtered_state,
            filtered_cov,
            predict_state,
            predict_cov,
            log_marginal_likelihood)


# Sample covariance. Handles differing shapes.
def _weight_covariance(x, y=None, weights_mean=None, weights_cov=None, need_mean=True):
    """Weighted covariance, assuming samples are the leftmost axis."""
    x = tf.convert_to_tensor(x, name='x')
    # Covariance *only* uses the centered versions of x (and y).
    if weights_mean is not None and need_mean:
        x = x - tf.squeeze(tf.transpose(x) @ weights_mean, axis=-1)

    if y is None:
        y = x
    else:
        y = tf.convert_to_tensor(y, name='y', dtype=x.dtype)

    if weights_cov is not None:
        x = x * weights_cov

    return tf.reduce_sum(tf.linalg.matmul(
        x[..., tf.newaxis],
        y[..., tf.newaxis], adjoint_b=True), axis=0)


def _sigma_samples(mean, cov, sqrt_sigma, col_val):
    """Sample from sigma points, for UKF.
    Args:
        mean: estimated mean from last time
        cov: estimated covariance from last time
        sqrt_sigma: square root of sigma point
        col_val: column index

    Returns:
    """

    def case_0():
        return mean

    def case_1():
        return mean + sqrt_sigma * cov[:, col_val - 1]

    def case_2():
        return mean - sqrt_sigma * cov[:, col_val - mean.shape[0] - 1]

    result = tf.cond(tf.equal(col_val, 0), case_0,
                     lambda: tf.cond(
                         tf.logical_and(tf.greater_equal(col_val, 1), tf.less_equal(col_val, mean.shape[0])),
                         case_1,
                         lambda: tf.cond(tf.logical_and(tf.greater_equal(col_val, mean.shape[0] + 1),
                                                        tf.less_equal(col_val, 2 * mean.shape[0])), case_2,
                                         case_0)))
    return result