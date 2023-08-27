"""
Approximation the State Space Model
"""
from Inference.Kalman.extended_kalman_filter import extended_kalman_filter as ekf
from Inference.Kalman.kalman_smoother import kalman_smoother as ks
from Inference.Kalman.kalman_filter import kalman_filter as kf
from Inference.Kalman.marginal_likelihood import marginal_likelihood as cal_likelihood
import tensorflow_probability.python.internal.prefer_static as ps
import tensorflow as tf
from Models.ssm_nlg import NonlinearSSM


def approximate_model(ssm_model,
                      observations,
                      max_iter=100,
                      conv_tol=1e-8,
                      abs_tol=1e-4
                      ):
    """Approximation the State Space Model
        1. Approximation of nonlinear Gaussian -> linear Gaussian: extended Kalman smoother
        2. Approximation of linear non-Gaussian -> linear Gaussian: Laplace Approximation
        3. Approximation of nonlinear non-Gaussian -> ?
    Currently this method approximates the nonlinear model to linear one using extended Kalman smoother
    Args:
        observations:
        max_iter:
        abs_tol:
        ssm_model: nonlinear Gaussian state space model
        conv_tol: Positive tolerance parameter used in Gaussian approximation.
    Returns:
        approximate_ssm: approximate linear Gaussian state space model
    """
    max_iter = tf.convert_to_tensor(max_iter, dtype=tf.int32, name="maximum_iteration")
    conv_tol = tf.convert_to_tensor(conv_tol, dtype=observations.dtype, name="signed_tolerance")
    abs_tol = tf.convert_to_tensor(abs_tol, dtype=observations.dtype, name="absolute_tolerance")

    approx_lg_ssm = initial_mode_by_ekf(ssm_model=ssm_model, observations=observations)
    init_lg_latent_estimate, _ = ks(approx_lg_ssm, observations)
    init_rel_diff = tf.constant(1e3, dtype=observations.dtype, name="relative_difference")
    init_abs_diff = tf.constant(1., dtype=observations.dtype, name="aboslute_difference")
    init_idx = tf.constant(0, dtype=tf.int32, name="iteration")
    init_log_likelihood = kf(approx_lg_ssm, observations, log_likelihood=True)

    def _loop_condition(idx, rel_diff, abs_diff, log_likelihood, lg_latent_estimate):
        return tf.logical_and(
            tf.logical_and(
                tf.less(idx, max_iter),
                tf.greater(rel_diff, conv_tol)),
            tf.greater(abs_diff, abs_tol))

    def _loop_body(idx, rel_diff, abs_diff, log_likelihood, lg_latent_estimate):
        idx += 1
        updated_ssm = lg_latent_update(ssm_model=ssm_model,
                                                       observations=observations,
                                                       lg_latent_estimate=lg_latent_estimate)
        lg_latent_estimate_new, _ = ks(updated_ssm, observations)
        new_likelihood = cal_likelihood(ssm_model=updated_ssm,
                                        observations=observations,
                                        latent_states=lg_latent_estimate_new)

        abs_diff = new_likelihood - log_likelihood
        rel_diff = abs_diff / tf.abs(log_likelihood)

        # go to far with previous mode_estimate, backtrack between mode_estimate_old and mode_estimate
        if rel_diff < -conv_tol and abs_diff > tf.constant(1e-4):
            rel_diff, abs_diff, \
                new_likelihood, lg_latent_estimate_new = _traceback_correction(updated_ssm, observations, rel_diff, abs_diff, log_likelihood, lg_latent_estimate,
                                  lg_latent_estimate_new, abs_tol, conv_tol)

        return idx, rel_diff, abs_diff, new_likelihood, lg_latent_estimate_new

    tf.while_loop(cond=_loop_condition,
                  body=_loop_body,
                  loop_vars=[init_rel_diff, init_abs_diff, init_idx, init_log_likelihood, init_lg_latent_estimate],
                  shape_invariants=[init_rel_diff.shape,
                                    init_abs_diff.shape,
                                    init_idx.shape,
                                    init_log_likelihood.shape,
                                    init_lg_latent_estimate.shape])


def _traceback_correction(updated_ssm, observations, rel_diff, abs_diff, log_likelihood, lg_latent_estimate,
                          lg_latent_estimate_new, abs_tol, conv_tol):
    """acktrack between mode_estimate_old and mode_estimate
    """
    init_inner_idx = tf.constant(0, dtype=tf.int32, name="iteration")
    init_step_size = tf.constant(1.0, dtype=rel_diff.dtype, name="step_size")

    def _condition(step_size, rel_diff, abs_diff, inner_idx, log_likelihood, lg_mode_estimate):
        return tf.logical_and(
            tf.logical_and(
                tf.less(inner_idx, tf.convert_to_tensor(10, dtype=tf.int32)),
                tf.greater(rel_diff, conv_tol)),
            tf.greater(abs_diff, abs_tol))

    def _correct_step(step_size, rel_diff, abs_diff, inner_idx, log_likelihood, lg_mode_estimate):
        inner_idx += 1
        step_size /= 2.0
        lg_mode_estimate_adapt = (1.0 - step_size)*lg_latent_estimate + step_size*lg_latent_estimate_new
        new_likelihood = cal_likelihood(ssm_model=updated_ssm,
                                        observations=observations,
                                        latent_states=lg_mode_estimate_adapt)
        abs_diff = new_likelihood - log_likelihood
        rel_diff = abs_diff / tf.abs(log_likelihood)

        return step_size, rel_diff, abs_diff, inner_idx, new_likelihood, lg_mode_estimate_adapt

    step_size, rel_diff, abs_diff, inner_idx, \
        new_likelihood, lg_mode_estimate_adapt = tf.while_loop(cond=_condition,
                         body=_correct_step,
                         loop_vars=[init_step_size, rel_diff, abs_diff, init_inner_idx, log_likelihood, lg_latent_estimate])
    return rel_diff, abs_diff, new_likelihood, lg_mode_estimate_adapt

def _generate_lg_mtx(ssm_model, filtered_means,
                     predicted_means, time_steps):
    def _convert_para(args):
        filtered_mean = args[0]
        predicted_mean = args[1]
        time_step = args[2]

        approx_state_mtx = ssm_model.transition_fn_grad(time_step, filtered_mean)
        approx_obs_mtx = ssm_model.observation_fn_grad(time_step, predicted_mean)

        current_obs_noise = ssm_model.observation_dist(time_step, predicted_mean).covariance()
        current_state_noise = ssm_model.transition_dist(time_step, filtered_mean).covariance()

        approx_state_input = ssm_model.transition_dist(time_step, filtered_mean).mean() \
                             - tf.matmul(approx_state_mtx, filtered_mean)
        approx_obs_input = ssm_model.observation_dist(time_step, predicted_mean).mean() \
                           - tf.matmul(approx_obs_mtx, predicted_mean)

        return approx_state_mtx, approx_obs_mtx, current_obs_noise, \
            current_state_noise, approx_state_input, approx_obs_input

    return tf.vectorized_map(_convert_para,
                             [filtered_means, predicted_means, time_steps])


def lg_latent_update(ssm_model,
                     observations,
                     lg_latent_estimate):
    """
    Args:
        observations: observated data
        ssm_model: original ssm
        lg_latent_estimate: estimated linear gaussian mean
    Returns:
        log marginal likelihood of the linear Gaussian model
    """

    approx_state_mtx, approx_obs_mtx, obs_noise, \
        state_noise, approx_state_input, approx_obs_input = _generate_lg_mtx(ssm_model,
                                                                             lg_latent_estimate,
                                                                             lg_latent_estimate,
                                                                             ssm_model.num_timesteps)
    approx_ssm = NonlinearSSM.create_model(num_timesteps=ssm_model.num_timesteps,
                                           observation_size=ssm_model.num_timesteps,
                                           latent_size=ssm_model.num_timesteps,
                                           initial_state_mean=ssm_model.initial_state_prior.mean(),
                                           initial_state_cov=ssm_model.initial_state_prior.covariance(),
                                           state_noise_std=tf.transpose(state_noise, perm=[1, 2, 0]),
                                           obs_noise_std=tf.transpose(obs_noise, perm=[1, 2, 0]),
                                           transition_matrix=tf.transpose(approx_state_mtx, perm=[1, 2, 0]),
                                           observation_matrix=tf.transpose(approx_obs_mtx, perm=[1, 2, 0]),
                                           input_state=tf.transpose(approx_state_input, perm=[1, 2, 0]),
                                           input_obs=tf.transpose(approx_obs_input, perm=[1, 2, 0]),
                                           nonlinear_type="linear_gaussian")
    return approx_ssm


def initial_mode_by_ekf(ssm_model,
                        observations):
    (filtered_means, filtered_covs,
     predicted_means, predicted_covs, _) = ekf(ssm_model, observations)

    predicted_means = predicted_means[1:, ...]
    time_steps = ssm_model.num_timesteps
    obs_dim = ps.shape(observations)[-1]
    state_dim = ssm_model.state_dim

    approx_state_mtx, approx_obs_mtx, obs_noise, \
        state_noise, approx_state_input, approx_obs_input = _generate_lg_mtx(ssm_model,
                                                                             filtered_means,
                                                                             predicted_means,
                                                                             time_steps)

    approx_ssm = NonlinearSSM.create_model(num_timesteps=time_steps,
                                           observation_size=obs_dim,
                                           latent_size=state_dim,
                                           initial_state_mean=ssm_model.initial_state_prior.mean(),
                                           initial_state_cov=ssm_model.initial_state_prior.covariance(),
                                           state_noise_std=tf.transpose(state_noise, perm=[1, 2, 0]),
                                           obs_noise_std=tf.transpose(obs_noise, perm=[1, 2, 0]),
                                           transition_matrix=tf.transpose(approx_state_mtx, perm=[1, 2, 0]),
                                           observation_matrix=tf.transpose(approx_obs_mtx, perm=[1, 2, 0]),
                                           input_state=tf.transpose(approx_state_input, perm=[1, 2, 0]),
                                           input_obs=tf.transpose(approx_obs_input, perm=[1, 2, 0]),
                                           nonlinear_type="linear_gaussian")
    return approx_ssm
