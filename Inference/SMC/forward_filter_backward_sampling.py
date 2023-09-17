import tensorflow as tf
from collections import namedtuple
from tensorflow_probability.python.internal import samplers
from Inference.SMC import resampling as resample
from .bootstrap_particle_filter import bootstrap_particle_filter
from .auxiliary_particle_filter import auxiliary_particle_filter
from .extend_kalman_particle_filter import extended_kalman_particle_filter
from Utils.smc_utils.smc_utils import posterior_mean_var
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import distribution_util as dist_util
import tensorflow_probability as tfp
tfd = tfp.distributions

return_results = namedtuple(
    'InferTrajectories', ['trajectories', 'incremental_log_marginal_likelihoods',
                          'smoother_mean'
                          ])

# TODO: rewrite, reduce the nums of move_dims

def forward_filter_backward_sampling(ssm_model,
                                     observations,
                                     num_particles,
                                     particle_filter_name,
                                     initial_state_proposal=None,
                                     resample_ess=0.5,
                                     resample_fn='systematic',
                                     seed=None,
                                     conditional_sample=None,
                                     is_conditional=False,
                                     is_guided=False,
                                     back_trajectory_num=None,
                                     is_one_trajectory=False):  # pylint: disable=g-doc-args
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
    with tf.name_scope('forward_filter_backward_smoother'):
        pf_seed, resample_seed = samplers.split_seed(
            seed, salt='backward_sampling')

        num_time_step = tf.get_static_value(tf.shape(observations))[0]
        if particle_filter_name == 'ekf':
            infer_result = extended_kalman_particle_filter(ssm_model=ssm_model,
                                                           observations=observations,
                                                           resample_fn=resample_fn,
                                                           resample_ess=resample_ess,
                                                           conditional_sample=conditional_sample,
                                                           is_conditional=is_conditional,
                                                           initial_state_proposal=initial_state_proposal,
                                                           num_particles=num_particles)
        elif particle_filter_name == 'bsf':
            infer_result = bootstrap_particle_filter(ssm_model=ssm_model,
                                                     resample_fn=resample_fn,
                                                     observations=observations,
                                                     resample_ess=resample_ess,
                                                     conditional_sample=conditional_sample,
                                                     is_conditional=is_conditional,
                                                     initial_state_proposal=initial_state_proposal,
                                                     num_particles=num_particles)
        elif particle_filter_name == 'apf':
            infer_result = auxiliary_particle_filter(ssm_model=ssm_model,
                                                     resample_fn=resample_fn,
                                                     observations=observations,
                                                     conditional_sample=conditional_sample,
                                                     is_conditional=is_conditional,
                                                     resample_ess=resample_ess,
                                                     initial_state_proposal=initial_state_proposal,
                                                     num_particles=num_particles,
                                                     is_guided=is_guided)
        else:
            raise NotImplementedError('No particle method')

        particles = infer_result.particles
        log_weights = infer_result.log_weights
        incremental_log_marginal_likelihoods = infer_result.incremental_log_marginal_likelihoods
        all_time_step = tf.get_static_value(num_time_step - 2)
        next_step_particles = tf.roll(particles, shift=-1, axis=0)

        if back_trajectory_num is None:
            back_trajectory_num = num_particles
        if is_one_trajectory:
            back_trajectory_num = 1

        idx_last_resample = _init_backward_sampling(log_weights[-1], trajectory_num = back_trajectory_num, seed=resample_seed)
        # tf.print(f" fuck {idx_last_resample}")

        backward_fn = _backward_sampling_step(transition_fn=ssm_model.transition_dist,
                                            trajectory_num=back_trajectory_num, seed=seed)
        backward_indx, _ = tf.scan(backward_fn,
                                   elems=(log_weights[:-1], particles[:-1], next_step_particles[:-1]),
                                   initializer=(idx_last_resample,
                                                all_time_step),
                                   reverse=True)
        backward_indx = tf.concat([backward_indx, [idx_last_resample]], axis=0)

        trajectories = _output_backward_sampling(particles, backward_indx)

        trajectories_move = dist_util.move_dimension(trajectories, source_idx=-1, dest_idx=1)
        smoother_mean, _, _, _ = posterior_mean_var(trajectories_move, log_weights,
                                                    num_time_step)

        return return_results(trajectories=trajectories,
                              incremental_log_marginal_likelihoods=incremental_log_marginal_likelihoods,
                              smoother_mean=smoother_mean)


def _init_backward_sampling(weight_last_step, trajectory_num, seed=None):

    extracted_sample_idx = resample._resample_multinomial(weight_last_step, resample_num=trajectory_num, seed=seed)

    # weights_last = tf.nn.softmax(weight_last_step, axis=0)
    #
    # # batch multinomial
    # weights_move = dist_util.move_dimension(weights_last, source_idx=0, dest_idx=-1)
    # extracted_sample_idx = tfd.Categorical(probs=weights_move).sample(trajectory_num, seed=seed)
    # trajectory_num, batch

    return dist_util.move_dimension(extracted_sample_idx, source_idx=-1, dest_idx=0)


def _backward_sampling_step(transition_fn, trajectory_num, seed=None):
    with tf.name_scope('backward_sampling_step'):
        def _one_step(back_sample_index, weights_and_particles):
            # trajectory_num, (chain)
            next_index, time_step = back_sample_index
            current_forward_log_weights, \
                current_step_particles, next_step_particles = weights_and_particles

            # have more than 1 chains
            next_step_particles_move = dist_util.move_dimension(next_step_particles, source_idx=-1, dest_idx=1)
            new_shape = tf.concat([ps.shape(next_step_particles_move)[:2], [-1]], axis=0)
            unflatten_shape = tf.concat([[trajectory_num],
                                         ps.shape(next_step_particles_move)[2:]], axis=0)
            next_step_particles_flatten = tf.reshape(next_step_particles_move, shape=new_shape)
            # move the multiple chains to the batch dimensions
            next_step_particles_flatten = dist_util.move_dimension(next_step_particles_flatten, source_idx=-1, dest_idx=0)
            """flatten and unflatten because chains may have multiple dims
            """
            next_step_particles_select = tf.vectorized_map(_gather, [next_step_particles_flatten, next_index])
            # move back
            next_step_particles_select = dist_util.move_dimension(next_step_particles_select, source_idx=0, dest_idx=-1)
            # unflatten -> (chain), trajectory_num
            next_step_particles_select = tf.reshape(next_step_particles_select, unflatten_shape)

            # transition_move = transition_fn(time_step, current_step_particles).log_prob(next_step_particles_select)
            # trajectory_num, particles, (chain)
            transition_move = tf.vectorized_map(lambda x:
                                                transition_fn(time_step, current_step_particles).log_prob(x[..., tf.newaxis]),
                                                next_step_particles_select)
            updated_weights = tf.nn.softmax(transition_move + current_forward_log_weights, axis=1)

            # batch multinomial
            weights_move = dist_util.move_dimension(updated_weights, source_idx=1, dest_idx=-1)
            current_index = tfd.Categorical(probs=weights_move).sample(seed=seed)
            # tf.print(f"sum? {tf.reduce_sum(weights_move, axis=-1)}")
            # tf.print(f"problem? {current_index}")

            return (dist_util.move_dimension(current_index, source_idx=0, dest_idx=-1),
                    time_step - 1)

    return _one_step


def _output_backward_sampling(particles, moved_indices):
    # particles: time_steps, num_particles, (chains), state_dim
    # moved_indices: time_steps, (chains), trajectory_num

    #  time_steps, (chains), state_dim, num_particles
    particles_move = dist_util.move_dimension(particles, source_idx=1, dest_idx=-1)

    def _gather_with_gather(args):
        particle = args[0]
        indx = args[1]
        def _gather_inside(args):
            return tf.gather(args[0], args[1], axis=-1)
        return tf.vectorized_map(_gather_inside, [particle, indx])

    trajectory = tf.vectorized_map(_gather_with_gather, [particles_move, moved_indices])
    new_shape = tf.concat([ps.shape(particles_move)[:2], [-1]], axis=0)
    return tf.reshape(trajectory, new_shape)


def _gather(args):
    return tf.gather(args[0], args[1], axis=0)