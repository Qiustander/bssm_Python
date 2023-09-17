"""Weighted resampling methods, e.g., for use in SMC methods."""

import tensorflow as tf
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import distribution_util as dist_util

__all__ = [
    'resample',
    '_resample_stratified',
    '_resample_systematic',
    '_resample_residual',
    '_resample_multinomial',
]


def resample(particles,
             log_weights,
             resample_fn,
             target_log_weights=None,
             is_conditional=False,
             seed=None):
    """Resamples the current particles according to provided weights.

  Args:
    particles: Nested structure of `Tensor`s each of shape
      `[num_particles, b1, ..., bN, ...]`, where
      `b1, ..., bN` are optional batch dimensions.
    log_weights: float `Tensor` of shape `[num_particles, b1, ..., bN]`, where
      `b1, ..., bN` are optional batch dimensions.
    resample_fn: choose the function used for resampling.
      Use 'resample_independent' for independent resamples.
      Use 'resample_stratified' for stratified resampling.
      Use 'resample_systematic' for systematic resampling.
    target_log_weights: optional float `Tensor` of the same shape and dtype as
      `log_weights`, specifying the target measure on `particles` if this is
      different from that implied by normalizing `log_weights`. The
      returned `log_weights_after_resampling` will represent this measure. If
      `None`, the target measure is implicitly taken to be the normalized
      log weights (`log_weights - tf.reduce_logsumexp(log_weights, axis=0)`).
      Default value: `None`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

  Returns:
    resampled_particles: Nested structure of `Tensor`s, matching `particles`.
    resample_indices: int `Tensor` of shape `[num_particles, b1, ..., bN]`.
    log_weights_after_resampling: float `Tensor` of same shape and dtype as
      `log_weights`, such that weighted sums of the resampled particles are
      equal (in expectation over the resampling step) to weighted sums of
      the original particles:
      `E [ exp(log_weights_after_resampling) * some_fn(resampled_particles) ]
      == exp(target_log_weights) * some_fn(particles)`.
      If no `target_log_weights` was specified, the log weights after
      resampling are uniformly equal to `-log(num_particles)`.
  """
    with tf.name_scope('resample'):
        num_particles = ps.size0(log_weights)

        # Normalize the weights and sample the ancestral indices.
        log_probs = tf.math.log_softmax(log_weights, axis=0)
        if is_conditional:
            # only sample N-1 particles
            resample_indices = resample_fn(log_probs, num_particles - 1, seed=seed)
            resampled_indices = tf.concat([tf.zeros([1, *ps.shape(resample_indices)[1:]],
                                                           dtype=resample_indices.dtype),
                                                  resample_indices],
                                                 axis=0)
        else:
            resampled_indices = resample_fn(log_probs, num_particles, seed=seed)

        gather_ancestors = lambda x: (  # pylint: disable=g-long-lambda
            mcmc_util.index_remapping_gather(x, resampled_indices, axis=0))
        resampled_particles = tf.nest.map_structure(gather_ancestors, particles)
        if target_log_weights is None:
            log_weights_after_resampling = tf.fill(ps.shape(log_weights),
                                                   -0.)
        else:
            importance_weights = target_log_weights - log_weights
            log_weights_after_resampling = tf.nest.map_structure(
                gather_ancestors, importance_weights)
    return resampled_particles, resampled_indices, log_weights_after_resampling


def _resample_residual(weights, resample_num, seed=None, name=None):
    """ Performs the residual resampling algorithm used by particle filters.

    Based on observation that we don't need to use random numbers to select
    most of the weights. Take int(N*w^i) samples of each particle i, and then
    resample any remaining using a standard resampling algorithm [1]


    Parameters
    ----------

    weights : tf.tensor, not logarithm, with shape [b, N]
    resample_num: resampling particles, default is weights.shape[-1]
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: tf name scope

    Returns
    -------

    resample_index : tf arrary of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.

    References
    ----------

    .. [1] J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
       systems. Journal of the American Statistical Association,
       93(443):1032–1044, 1998.
    """
    with tf.name_scope(name or 'resample_resiudual'):
        # TODO: batch based
        weights = tf.math.log_softmax(weights, axis=0)
        weights = tf.exp(tf.convert_to_tensor(weights, dtype_hint=tf.float32))
        if not resample_num:
            resample_num = ps.shape(weights)[0]

        # deterministic sampling
        weights_int = resample_num * weights
        floor_weight = tf.floor(weights_int)
        res_weight = weights - floor_weight

        # deal with interger part
        range_indx = tf.range(ps.shape(weights)[0])
        int_weight = tf.repeat(range_indx,
                                 tf.cast(floor_weight, dtype=range_indx.dtype))

        if tf.size(int_weight) == resample_num:
            return int_weight
        else:
            cdf_res_weights = tf.concat([tf.math.cumsum(res_weight/tf.reduce_sum(res_weight), axis=-1)[..., :-1],
                                     tf.ones([1, ], dtype=weights.dtype)],
                                    axis=-1)
            searchpoints = uniform.Uniform(low=ps.cast(0., dtype=weights.dtype),
                                           high=ps.cast(1., dtype=weights.dtype)).\
                sample(resample_num - tf.size(int_weight), seed=seed)
            int_weight = tf.concat([int_weight,
                                    tf.searchsorted(cdf_res_weights, searchpoints)], axis=-1)
            return tf.sort(int_weight)


def _resample_stratified(weights, resample_num, seed=None, name=None):
    """ Performs the stratified resampling algorithm used by particle filters.

    This algorithms aims to make selections relatively uniformly across the
    particles. It divides the cumulative sum of the weights into N equal
    divisions, and then selects one particle randomly from each division. This
    guarantees that each sample is between 0 and 2/N apart.

    Parameters
    ----------
    weights : tf.tensor, not logarithm, with shape [b, N]
    resample_num: resampling particles, default is weights.shape[-1]
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: tf name scope

    Returns
    -------

    resample_index : tf arrary of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    with tf.name_scope(name or 'resample_stratified'):
        # normalize in case there is direct usage
        weights = tf.math.log_softmax(weights, axis=0)

        weights = tf.exp(tf.convert_to_tensor(weights, dtype_hint=tf.float32))

        if not resample_num:
            resample_num = ps.shape(weights)[0]
        batch_shape = ps.shape(weights)[1:]
        points_shape = ps.concat([[resample_num],
                                  batch_shape], axis=0)
        full_prob_shape = ps.concat([[1],
                                     batch_shape], axis=0)

        # Draw an offset for every element of an event, with size [b, Ns]
        interval_width = ps.cast(1. / resample_num, dtype=weights.dtype)
        offsets = uniform.Uniform(low=ps.cast(0., dtype=weights.dtype),
                                  high=interval_width).sample(
            points_shape, seed=seed)

        # The unit interval is divided into equal partitions and each point
        # is a random offset into a partition.
        resampling_space = tf.linspace(
            start=tf.broadcast_to(ps.cast(0., dtype=weights.dtype),
                                  batch_shape),
            stop=1 - interval_width,
            num=resample_num) + offsets

        # Resampling
        cdf_weights = tf.concat([tf.math.cumsum(weights, axis=0)[:-1],
                                 tf.broadcast_to(tf.constant(1., dtype=weights.dtype),
                                 full_prob_shape)],
                                axis=0)
        # tf.searchsorted works for innermost dimension, so need to move the dimension first
        cdf_weights_flatten = tf.reshape(cdf_weights, shape=[ps.size0(cdf_weights), -1])
        cdf_weights_flatten = dist_util.move_dimension(cdf_weights_flatten, source_idx=0, dest_idx=-1)
        resampling_space_flatten = tf.reshape(resampling_space, shape=[ps.size0(resampling_space), -1])
        resampling_space_flatten = dist_util.move_dimension(resampling_space_flatten, source_idx=0, dest_idx=-1)

        def _search_sort(args):
            return tf.searchsorted(args[0], args[1])
        resample_index = tf.vectorized_map(_search_sort, [cdf_weights_flatten, resampling_space_flatten])
        resample_index = dist_util.move_dimension(resample_index, source_idx=0, dest_idx=-1)

        return tf.reshape(resample_index, points_shape)


def _resample_systematic(weights, resample_num, seed=None, name=None):
    """ Performs the systemic resampling algorithm used by particle filters.

    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.

    Parameters
    ----------
    weights : tf.tensor, not logarithm, with shape [b, N]
    resample_num: resampling particles, default is weights.shape[-1]
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: tf name scope

    Returns
    -------

    resample_index : tf arrary of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    with tf.name_scope(name or 'resample_systematic'):
        # normalize in case there is direct usage
        weights = tf.math.log_softmax(weights, axis=0)

        weights = tf.exp(tf.convert_to_tensor(weights, dtype_hint=tf.float32))

        if not resample_num:
            resample_num = ps.shape(weights)[0]

        batch_shape = ps.shape(weights)[1:]
        points_shape = ps.concat([[resample_num],
                                  batch_shape], axis=0)
        full_prob_shape = ps.concat([[1],
                                     batch_shape], axis=0)

        # Draw an offset for whole events, with size [b, ]
        interval_width = ps.cast(1. / resample_num, dtype=weights.dtype)
        offsets = uniform.Uniform(low=ps.cast(0., dtype=weights.dtype),
                                  high=interval_width).sample(seed=seed)

        # The unit interval is divided into equal partitions and each point
        # is a random offset into a partition.
        resampling_space = tf.linspace(
            start=tf.broadcast_to(ps.cast(0., dtype=weights.dtype),
                              batch_shape),
            stop=1 - interval_width,
            num=resample_num) + offsets

        # Resampling
        cdf_weights = tf.concat([tf.math.cumsum(weights, axis=0)[:-1],
                                 tf.broadcast_to(tf.constant(1., dtype=weights.dtype),
                                                 full_prob_shape)],
                                axis=0)
        # tf.searchsorted works for innermost dimension, so need to move the dimension first
        cdf_weights_flatten = tf.reshape(cdf_weights, shape=[ps.size0(cdf_weights), -1])
        cdf_weights_flatten = dist_util.move_dimension(cdf_weights_flatten, source_idx=0, dest_idx=-1)
        resampling_space_flatten = tf.reshape(resampling_space, shape=[ps.size0(resampling_space), -1])
        resampling_space_flatten = dist_util.move_dimension(resampling_space_flatten, source_idx=0, dest_idx=-1)

        def _search_sort(args):
            return tf.searchsorted(args[0], args[1])

        resample_index = tf.vectorized_map(_search_sort, [cdf_weights_flatten, resampling_space_flatten])
        resample_index = dist_util.move_dimension(resample_index, source_idx=0, dest_idx=-1)

        return tf.reshape(resample_index, points_shape)


def _resample_multinomial(weights, resample_num, seed=None, name=None):
    """ This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the
    resampled point based on a uniformly distributed random number. Run time
    is O(n log n). You do not want to use this algorithm in practice; for some
    reason it is popular in blogs and online courses so I included it for
    reference.

   Parameters
   ----------

    weights : tf.tensor, logarithm, with shape [b, N]
    resample_num: resampling particles, default is weights.shape[-1]
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: tf name scope

    Returns
    -------

    resample_index : tf arrary of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    with tf.name_scope(name or 'resample_multinomial'):
        # normalize in case there is direct usage
        weights = tf.math.log_softmax(weights, axis=0)

        weights = tf.exp(tf.convert_to_tensor(weights, dtype_hint=tf.float32))

        if not resample_num:
            resample_num = ps.shape(weights)[0]
        batch_shape = ps.shape(weights)[1:]
        points_shape = ps.concat([[resample_num],
                                  batch_shape], axis=0)
        # concat prob 1
        full_prob_shape = ps.concat([[1],
                                     batch_shape], axis=0)

        searchpoints = tf.sort(uniform.Uniform(low=ps.cast(0., dtype=weights.dtype),
                                       high=ps.cast(1., dtype=weights.dtype)).sample(points_shape, seed=seed), axis=0)

        # Resampling
        cdf_weights = tf.concat([tf.math.cumsum(weights, axis=0)[:-1],
                                 tf.broadcast_to(tf.constant(1., dtype=weights.dtype),
                                                 full_prob_shape)],
                                axis=0)

        # tf.searchsorted works for innermost dimension, so need to move the dimension first
        cdf_weights_flatten = tf.reshape(cdf_weights, shape=[ps.size0(cdf_weights), -1])
        cdf_weights_flatten = dist_util.move_dimension(cdf_weights_flatten, source_idx=0, dest_idx=-1)
        resampling_space_flatten = tf.reshape(searchpoints, shape=[ps.size0(searchpoints), -1])
        resampling_space_flatten = dist_util.move_dimension(resampling_space_flatten, source_idx=0, dest_idx=-1)

        def _search_sort(args):
            return tf.searchsorted(args[0], args[1])

        resample_index = tf.vectorized_map(_search_sort, [cdf_weights_flatten, resampling_space_flatten])
        resample_index = dist_util.move_dimension(resample_index, source_idx=0, dest_idx=-1)

        return tf.reshape(resample_index, points_shape)
