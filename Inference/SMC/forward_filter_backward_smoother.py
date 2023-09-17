import tensorflow as tf
from collections import namedtuple
from Utils.smc_utils.smc_utils import posterior_mean_var
from tensorflow_probability.python.internal import samplers

return_results = namedtuple(
    'ForwardFilterBackwardSmoother', ['filtered_mean', 'predicted_mean',
                                      'filtered_variance', 'predicted_variance', 'smoother_mean',
                                      'incremental_log_marginal_likelihoods', 'particles',
                                      'log_weights', 'parent_indices', 'accumulated_log_marginal_likelihood'])


def forward_filter_backward_smoother(ssm_model,
                                     observations,
                                     num_particles,
                                     particle_filter_name,
                                     resample_ess=0.5,
                                     resample_fn='systematic',
                                     seed=None,
                                     name=None):  # pylint: disable=g-doc-args
    """Forward Filter Backward Smoother Algorithm

     Estimate the full-path posterior distribution p(x_{t} | y_{0:T}) with simple
     tracing.

  ${particle_filter_arg_str}
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name for ops created by this method.
      Default value: `None` (i.e., `'infer_trajectories'`).
  Returns:

  #### References

  [1] Kitagawa, G (1996). Monte Carlo filter and smoother for non-Gaussian
     nonlinear state space models.
     Journal of Computational and Graphical Statistics, 5, 1-25.
     https://doi.org/10.2307/1390750

  """
    with tf.name_scope(name or 'forward_filter_backward_smoother'):
        if seed is None:
            seed = samplers.sanitize_seed(seed, name='forward_filter_backward_smoother')
        pf_seed, _ = samplers.split_seed(
            seed, salt='forward_filter_backward_smoother')

        num_time_step = tf.get_static_value(tf.shape(observations))[0]
        if particle_filter_name == 'ekf':
            from .extend_kalman_particle_filter import extended_kalman_particle_filter
            infer_result = extended_kalman_particle_filter(ssm_model=ssm_model,
                                                           observations=observations,
                                                           resample_fn=resample_fn,
                                                           resample_ess=resample_ess,
                                                           num_particles=num_particles,
                                                           seed=pf_seed)
        elif particle_filter_name == 'bsf':
            from .bootstrap_particle_filter import bootstrap_particle_filter
            infer_result = bootstrap_particle_filter(ssm_model=ssm_model,
                                                     resample_fn=resample_fn,
                                                     observations=observations,
                                                     resample_ess=resample_ess,
                                                     num_particles=num_particles,
                                                     seed=pf_seed)
        elif particle_filter_name == 'apf':
            from .auxiliary_particle_filter import auxiliary_particle_filter
            infer_result = auxiliary_particle_filter(ssm_model=ssm_model,
                                                     resample_fn=resample_fn,
                                                     observations=observations,
                                                     resample_ess=resample_ess,
                                                     num_particles=num_particles,
                                                     seed=pf_seed)
        else:
            raise NotImplementedError('No particle method')

        particles = infer_result.particles
        parent_indices = infer_result.parent_indices
        log_weights = infer_result.log_weights
        incremental_log_marginal_likelihoods = infer_result.incremental_log_marginal_likelihoods

        # Normalization of the log weights
        all_time_step = tf.get_static_value(num_time_step - 2)
        log_weights = tf.nn.log_softmax(log_weights, axis=1)
        next_particles = tf.roll(particles, shift=-1, axis=0)

        backward_smooth = _backward_filter_step(transition_fn=ssm_model.transition_dist)
        backward_weights, _ = tf.scan(backward_smooth,
                                      elems=(log_weights[:-1], particles[:-1], next_particles[:-1]),
                                      initializer=(log_weights[-1],
                                                   all_time_step),
                                      reverse=True)
        backward_weights = tf.concat([backward_weights, [log_weights[-1]]], axis=0)

        smoother_mean, _, _, _ = posterior_mean_var(particles, backward_weights,
                                                    num_time_step)

        return return_results(filtered_mean=infer_result.filtered_mean, predicted_mean=infer_result.predicted_mean,
                              smoother_mean=smoother_mean,
                              filtered_variance=infer_result.filtered_variance,
                              predicted_variance=infer_result.predicted_variance,
                              incremental_log_marginal_likelihoods=incremental_log_marginal_likelihoods,
                              accumulated_log_marginal_likelihood=infer_result.accumulated_log_marginal_likelihood,
                              particles=particles, log_weights=log_weights, parent_indices=parent_indices)


def _backward_filter_step(transition_fn):
    with tf.name_scope('backward_filter_step'):
        def _one_step(back_weights, weights_and_particles):
            backward_weights, time_step = back_weights
            current_forward_weights, \
                current_step_particles, next_step_particles = weights_and_particles
            current_step_particles = tf.squeeze(current_step_particles, axis=-1)
            next_step_particles = tf.squeeze(next_step_particles, axis=-1)

            # sum_k W_t^k f(X_{t+1}^j | X_t^k)
            def _deno_sum(x):
                # conduct the sum along the k, return element for each j
                transition_move = transition_fn(time_step, current_step_particles).log_prob(x)
                # return the log(sum(exp))) for later usage
                return tf.math.reduce_logsumexp(transition_move + current_forward_weights)

            denominator_log_sum = tf.vectorized_map(_deno_sum,
                                                next_step_particles)

            # sum_k W_{t+1|T}^j f(X_{t+1}^j | X_t^i)
            whole_sum_fn = _whole_sum(transition_fn=transition_fn,
                                      time_step=time_step,
                                      backward_weights=backward_weights,
                                      next_step_particles=next_step_particles,
                                      denominator_log_sum=denominator_log_sum)

            backward_weights = tf.vectorized_map(whole_sum_fn,
                                                 (current_step_particles, current_forward_weights))
            backward_weights = tf.nn.log_softmax(backward_weights)

            return (backward_weights,
                    time_step - 1)

    return _one_step


def _whole_sum(transition_fn,
               time_step,
               backward_weights,
               next_step_particles,
               denominator_log_sum):
    def _inner_wrap(inputs):
        current_step_particle, current_forward_weight = inputs
        current_step_particle = current_step_particle[tf.newaxis, ...]
        # conduct the sum along the j, return element for each i
        transition_move = transition_fn(time_step, current_step_particle).log_prob(next_step_particles[..., tf.newaxis])
        intermediate_output = tf.math.reduce_logsumexp(transition_move + backward_weights - denominator_log_sum)
        return intermediate_output + current_forward_weight

    return _inner_wrap
