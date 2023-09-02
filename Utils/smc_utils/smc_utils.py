from tensorflow_probability.python.internal import prefer_static as ps
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.math import linalg
import sys
tfd = tfp.distributions


def default_trace_fn(state, kernel_results):
    return (state.particles,
            state.log_weights,
            kernel_results.parent_indices,
            kernel_results.incremental_log_marginal_likelihood,
            kernel_results.accumulated_log_marginal_likelihood)


def posterior_mean_var(particles, log_weights, num_time_step):
    """

    Args:
        particles: the particles with shape [num_time_steps, num_particles, state_dim]
        log_weights: weights in the logarithms with the shape [num_time_steps, num_particles]
        num_time_step: number of time steps
    Returns:
        filtered_mean:
        filtered_variance:
        predicted_mean:
        predicted_variance:
    """

    # check the output particle filter is whether dict

    def _posteror_processing(input_particles):

        input_particles = tf.cond((ps.rank(input_particles) - ps.rank(log_weights))==0,
                                  lambda : input_particles[..., tf.newaxis],
                                  lambda : input_particles)

        weights = tf.nn.softmax(log_weights)[..., tf.newaxis]
        filtered_mean = tf.reduce_sum(weights * input_particles, axis=1)
        predicted_mean = tf.reduce_mean(input_particles, axis=1)

        predicted_variance = tf.einsum("...ij,...ik->...jk",
                                       input_particles - tf.expand_dims(predicted_mean, axis=1),
                                       input_particles - tf.expand_dims(predicted_mean, axis=1))
        predicted_variance /= num_time_step
        filtered_variance = tf.einsum("...ij,...ik->...jk",
                                      weights * (input_particles - tf.expand_dims(filtered_mean, axis=1)),
                                      input_particles - tf.expand_dims(filtered_mean, axis=1))
        return (filtered_mean, predicted_mean,
            filtered_variance, predicted_variance)

    if isinstance(particles, dict):
        filtered_mean = predicted_mean = filtered_variance = predicted_variance = {}
        for key, tensor in particles.items():
            filtered_mean[key], predicted_mean[key],\
             filtered_variance[key], predicted_variance[key] = _posteror_processing(tensor)
    else:
        (filtered_mean, predicted_mean,
         filtered_variance, predicted_variance) = _posteror_processing(particles)

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

    filter_mean, filter_covariance = _extended_kalman_initial_step(initial_mean, initial_cov, observation_fn_grad,
                                                observation_dist, observation)

    return tfd.MultivariateNormalTriL(
        loc=filter_mean,
        scale_tril=tf.linalg.cholesky(filter_covariance))


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
    filter_particles, filter_covariance = ekf_step_fn(particles)

    return tfd.MultivariateNormalTriL(
        loc=filter_particles,
        scale_tril=tf.linalg.cholesky(filter_covariance))


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


#TODO: could we combine the functions?
def _extended_kalman_initial_step(init_mean, init_cov, observation_fn_grad,
                               observation_fn, observation):

    predicted_cov = init_cov
    predicted_mean = init_mean

    observation_dist = observation_fn(0, predicted_mean)
    observation_mean = observation_dist.mean()
    observation_cov = observation_dist.covariance()

    predicted_jacobian = observation_fn_grad(0, predicted_mean)
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

