import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.math import linalg
from .extended_kalman_filter import extended_kalman_filter
from tensorflow_probability.python.internal import prefer_static as ps

tfd = tfp.distributions


@tf.function
def extended_kalman_smoother(ssm_model, observations):
    """ Conduct the extended Kalman Smoother
    Args:
        ssm_model: model object (nonlinear model)
        observations: observed time series
    Returns:
      smoothed_means: Means of the smoothed marginal distributions
        p(z[t] | x[1:T]), as a Tensor of shape
        `sample_shape(x) + batch_shape + [num_timesteps, latent_size]`,
        which is of the same shape as filtered_means.
      smoothed_covs: Covariances of the smoothed marginal distributions
        p(z[t] | x[1:T]), as a Tensor of shape
        `sample_shape(x) + batch_shape + [num_timesteps, latent_size,
        latent_size]`. which is of the same shape as filtered_covs.
    """

    (filtered_means, filtered_covs,
     predicted_means, predicted_covs, _,) = extended_kalman_filter(
        ssm_model,
        observations
    )
    # extract the 1:last predicted mean & covariance
    predicted_means = predicted_means[1:, ...]
    predicted_covs = predicted_covs[1:, ...]

    # The means are assumed to be vectors. Adding a dummy index to
    # ensure the `matmul` op working smoothly.

    filtered_means = filtered_means[..., tf.newaxis]
    predicted_means = predicted_means[..., tf.newaxis]

    smoothed_means, smoothed_covs = backward_smoothing_pass(ssm_model.transition_fn_grad,
                                                            filtered_means, filtered_covs,
                                                            predicted_means, predicted_covs)

    smoothed_means = distribution_util.move_dimension(
        smoothed_means[..., 0], 0, -2)

    return smoothed_means, smoothed_covs


def backward_smoothing_pass(transition_fn_grad, filtered_means, filtered_covs,
                            predicted_means, predicted_covs):
    """Run the backward pass in extended Kalman smoother.

    The backward smoothing is using Rauch, Tung and Striebel smoother as
    as discussed in section 18.3.2 of Kevin P. Murphy, 2012, Machine Learning:
    A Probabilistic Perspective, The MIT Press. The inputs are returned by
    kalman filter.

    Args:
      transition_fn_grad: callable derivative of the transition function
      filtered_means: Means of the per-timestep filtered marginal
        distributions p(z[t] | x[:t]), as a Tensor of shape
        `sample_shape(x) + batch_shape + [num_timesteps, latent_size]`.
      filtered_covs: Covariances of the per-timestep filtered marginal
        distributions p(z[t] | x[:t]), as a Tensor of shape
        `sample_shape(x) + batch_shape + [num_timesteps, latent_size,
        latent_size]`.
      predicted_means: Means of the per-timestep predictive
         distributions over latent states, p(z[t+1] | x[:t]), as a
         Tensor of shape `sample_shape(x) + batch_shape +
         [num_timesteps, latent_size]`.
      predicted_covs: Covariances of the per-timestep predictive
         distributions over latent states, p(z[t+1] | x[:t]), as a
         Tensor of shape `sample_shape(x) + batch_shape +
         [num_timesteps, latent_size, latent_size]`.

    Returns:
      smoothed_means
      smoothed_covs
    """
    update_step_fn = build_backward_pass_step(
        transition_fn_grad)

    initial_backward_mean = predicted_means[-1, ...]
    initial_backward_cov = predicted_covs[-1, ...]
    last_time_step = ps.size0(filtered_means)-1

    (smoothed_means, smoothed_covs, time_step) = tf.scan(update_step_fn,
                                              elems=(filtered_means,
                                                     filtered_covs,
                                                     predicted_means,
                                                     predicted_covs),
                                              initializer=(initial_backward_mean, initial_backward_cov, last_time_step),
                                              reverse=True)

    return smoothed_means, smoothed_covs


def build_backward_pass_step(transition_fn_grad):
    """Build a callable that perform one step for backward smoothing.

  Args:
    transition_fn_grad: callable taking the mu_t|t
      as `Tensor` argument, and returning a `LinearOperator`
      of shape `[latent_size, latent_size]`.

  Returns:
    backward_pass_step: a callable that updates a BackwardPassState
      from timestep `t` to `t-1`.
  """

    def backward_pass_step(state,
                           filtered_parameters):
        """Run a single step of backward smoothing."""

        (filtered_mean, filtered_cov,
         predicted_mean, predicted_cov) = filtered_parameters

        next_posterior_mean = state[0]
        next_posterior_cov = state[1]
        time_step = state[-1]

        posterior_mean, posterior_cov, time_step = backward_smoothing_update(
            filtered_mean,
            filtered_cov,
            predicted_mean,
            predicted_cov,
            next_posterior_mean,
            next_posterior_cov,
            transition_fn_grad,
            time_step)

        return posterior_mean, posterior_cov, time_step

    return backward_pass_step


def backward_smoothing_update(filtered_mean,
                              filtered_cov,
                              predicted_mean,
                              predicted_cov,
                              next_posterior_mean,
                              next_posterior_cov,
                              transition_fn_grad,
                              time_step):
    """Backward update for a Kalman smoother.

  Give the `filtered_mean` mu(t | t), `filtered_cov` sigma(t | t),
  `predicted_mean` mu(t+1 | t) and `predicted_cov` sigma(t+1 | t),
  as returns from the `forward_filter` function, as well as
  `next_posterior_mean` mu(t+1 | 1:T) and `next_posterior_cov` sigma(t+1 | 1:T),
  if the `transition_matrix` of states from time t to time t+1
  is given as A(t+1), the 1 step backward smoothed distribution parameter
  could be calculated as:
  p(z(t) | Obs(1:T)) = N( mu(t | 1:T), sigma(t | 1:T)),
  mu(t | 1:T) = mu(t | t) + J(t) * (mu(t+1 | 1:T) - mu(t+1 | t)),
  sigma(t | 1:T) = sigma(t | t)
                   + J(t) * (sigma(t+1 | 1:T) - sigma(t+1 | t) * J(t)',
  J(t) = sigma(t | t) * A(t+1)' / sigma(t+1 | t),
  where all the multiplications are matrix multiplication, and `/` is
  the matrix inverse. J(t) is the backward Kalman gain matrix.

  The algorithm can be intialized from mu(T | 1:T) and sigma(T | 1:T),
  which are the last step parameters returned by forward_filter.


  Args:
    filtered_mean: `Tensor` with event shape `[latent_size, 1]` and
      batch shape `B`, containing mu(t | t).
    filtered_cov: `Tensor` with event shape `[latent_size, latent_size]` and
      batch shape `B`, containing sigma(t | t).
    predicted_mean: `Tensor` with event shape `[latent_size, 1]` and
      batch shape `B`, containing mu(t+1 | t).
    predicted_cov: `Tensor` with event shape `[latent_size, latent_size]` and
      batch shape `B`, containing sigma(t+1 | t).
    next_posterior_mean: `Tensor` with event shape `[latent_size, 1]` and
      batch shape `B`, containing mu(t+1 | 1:T).
    next_posterior_cov: `Tensor` with event shape `[latent_size, latent_size]`
      and batch shape `B`, containing sigma(t+1 | 1:T).
    transition_fn_grad: `LinearOperator` with shape
      `[latent_size, latent_size]` and batch shape broadcastable
      to `B`.

  Returns:
    posterior_mean: `Tensor` with event shape `[latent_size, 1]` and
      batch shape `B`, containing mu(t | 1:T).
    posterior_cov: `Tensor` with event shape `[latent_size, latent_size]` and
      batch shape `B`, containing sigma(t | 1:T).
  """

    latent_size_is_static_and_scalar = (filtered_cov.shape[-2] == 1)

    # Compute backward Kalman gain:
    # J = F * T' * P^{-1}
    # Since both F(iltered) and P(redictive) are cov matrices,
    # thus self-adjoint, we can take the transpose.
    # computation:
    #      = (P^{-1} * T * F)'
    #      = (P^{-1} * tmp_gain_cov) '
    #      = (P \ tmp_gain_cov)'

    # reduce the dummy index to conduct the jacobian
    grad_mean = transition_fn_grad(time_step, tf.squeeze(filtered_mean, axis=-1))
    tmp_gain_cov = tf.linalg.LinearOperatorFullMatrix(grad_mean).matmul(filtered_cov)
    if latent_size_is_static_and_scalar:
        gain_transpose = tmp_gain_cov / predicted_cov
    else:
        gain_transpose = linalg.hpsd_solve(predicted_cov, tmp_gain_cov)

    posterior_mean = (filtered_mean +
                      tf.linalg.matmul(gain_transpose,
                                       next_posterior_mean - predicted_mean,
                                       adjoint_a=True))
    posterior_cov = (
            filtered_cov +
            tf.linalg.matmul(gain_transpose,
                             tf.linalg.matmul(
                                 next_posterior_cov - predicted_cov, gain_transpose),
                             adjoint_a=True))

    return posterior_mean, posterior_cov, time_step-1
