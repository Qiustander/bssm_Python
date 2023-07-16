from tensorflow_probability.python.internal import prefer_static as ps
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from tensorflow_probability.python.math import linalg


def proposal_fn(proposal_name):
    """
    Args:
        proposal_name: 'APF', 'Optimal'

    Returns: proposal function in the form of distribution

    """


def default_trace_fn(state, kernel_results):
    return (state.particles,
            state.log_weights,
            kernel_results.parent_indices,
            kernel_results.incremental_log_marginal_likelihood,
            kernel_results.accumulated_log_marginal_likelihood)


def posterior_mean_var(particles, log_weights):
    """

    Args:
        particles: the particles with shape [num_time_steps, num_particles, state_dim]
        log_weights: weights in the logarithms with the shape [num_time_steps, num_particles]
    Returns:
        filtered_mean:
        filtered_variance:
        predicted_mean:
        predicted_variance:
    """
    weights = tf.exp(log_weights)[..., tf.newaxis]
    filtered_mean = tf.reduce_sum(weights * particles, axis=1)
    predicted_mean = tf.reduce_mean(particles, axis=1)

    predicted_variance = tf.einsum("...ij,...ik->...jk",
                                   particles - predicted_mean[:, tf.newaxis],
                                   particles - predicted_mean[:, tf.newaxis]) / predicted_mean.shape[0]
    filtered_variance = tf.einsum("...ij,...ik->...jk",
                                  weights * (particles - filtered_mean[:, tf.newaxis]),
                                  particles - filtered_mean[:, tf.newaxis])

    return (filtered_mean, predicted_mean,
            filtered_variance, predicted_variance)


def extended_kalman_particle_initial(initial_prior_fn, observation,
                                  transition_dist, observation_dist,
                                  transition_fn_grad, observation_fn_grad):
    """
    extended kalman filter initial step with particles with batch based operation
    Args:
        initial_prior_fn:
        observation:
        transition_dist:
        observation_dist:
        transition_fn_grad:
        observation_fn_grad:

    Returns: Multivariate Normal Distribution

    """

    initial_mean = initial_prior_fn.mean()
    initial_cov = initial_prior_fn.covariance()

    ekf_step_fn = _extended_kalman_one_step(0, transition_fn_grad, observation_fn_grad,
                                            transition_dist, observation_dist, observation)

    filter_mean, filter_covariance = ekf_step_fn([initial_mean, initial_cov])

    return tfd.MultivariateNormalFullCovariance(
        loc=filter_mean,
        covariance_matrix=filter_covariance)


def extended_kalman_particle_step(step, particles, observation,
                                  transition_dist, observation_dist,
                                  transition_fn_grad, observation_fn_grad):
    """
    extended kalman filter update one step with particles with batch based operation
    Args:
        step: time step
        particles:
        transition_dist:
        observation_dist:
        transition_fn_grad:
        observation_fn_grad:

    Returns: Multivariate Normal Distribution

    """

    ekf_step_fn = _extended_kalman_one_step(step, transition_fn_grad, observation_fn_grad,
                                            transition_dist, observation_dist, observation)
    # TODO: revise to batch base, need revision in the nonlinear function type
    filter_particles, filter_covariance = tf.vectorized_map(ekf_step_fn, particles)
    # filter_particles, filter_covariance = ekf_step_fn(particles)

    return tfd.MultivariateNormalFullCovariance(
        loc=filter_particles,
        covariance_matrix=filter_covariance)


def _extended_kalman_one_step(step, transition_fn_grad, observation_fn_grad,
                              transition_fn, observation_fn, observation):
    def _one_step(particles):

        state_prior = transition_fn(step, particles)

        predicted_cov = state_prior.covariance()
        predicted_mean = state_prior.mean()

        observation_dist = observation_fn(step, predicted_mean)
        observation_mean = observation_dist.mean()
        observation_cov = observation_dist.covariance()

        predicted_jacobian = observation_fn_grad(step, predicted_mean)
        tmp_obs_cov = tf.matmul(predicted_jacobian, predicted_cov)
        residual_covariance = tf.matmul(
            predicted_jacobian, tmp_obs_cov, transpose_b=True) + observation_cov

        chol_residual_cov = tf.linalg.cholesky(residual_covariance)
        gain_transpose = linalg.hpsd_solve(
            residual_covariance, tmp_obs_cov, cholesky_matrix=chol_residual_cov)

        filtered_mean = predicted_mean + tf.matmul(
            gain_transpose,
            (observation - observation_mean)[..., tf.newaxis],
            transpose_a=True)[..., 0]

        tmp_term = -tf.matmul(predicted_jacobian, gain_transpose, transpose_a=True)
        tmp_term = tf.linalg.set_diag(tmp_term, tf.linalg.diag_part(tmp_term) + 1.)
        filtered_cov = (
                tf.matmul(
                    tmp_term, tf.matmul(predicted_cov, tmp_term), transpose_a=True) +
                tf.matmul(gain_transpose,
                          tf.matmul(observation_cov, gain_transpose), transpose_a=True))

        return filtered_mean, filtered_cov

    return _one_step
