"""Weighted resampling methods, e.g., for use in SMC methods."""

import tensorflow as tf
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.distributions import Multinomial
from tensorflow_probability.python.internal import prefer_static as ps

__all__ = [
    'resample',
    '_resample_stratified',
    '_resample_systematic',
    '_resample_residual',
    '_resample_multinomial',
]


def resample(method):
    """
    Args:
        method: resampling method, currently support residual, systematic, residual, multinomial
        seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
        weights: [b, N], batch of weights
        resample_num: default is the same as the weights
    Returns:
        resampled_idx: resampling index, leave the true resampling in outer function
    """
    resample_method = f"_resample_{method}"

    def resample_step(weights, resample_num=None, seed=None):
        with tf.name_scope('resample'):
            try:
                resampled_idx = globals()[f'{resample_method}'](weights, resample_num, seed)
            except:
                raise AssertionError(f'No {method} resampling is implemented!')
            return resampled_idx

    return resample_step


def _resample_residual(weights, resample_num, seed, name=None):
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

        weights = tf.convert_to_tensor(weights, dtype_hint=tf.float32)
        if not resample_num:
            resample_num = ps.shape(weights)[-1]

        #deterministic sampling
        weights_int = resample_num * weights
        floor_weight = tf.floor(weights_int)
        res_weight = weights_int - floor_weight

        # deal with interger part
        range_weight = tf.repeat(tf.range(ps.shape(weights)[-1])[tf.newaxis, ...], ps.shape(weights)[0], axis=-1)

        pass
        # TODO: finish it


def _resample_stratified(weights, resample_num, seed, name=None):
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
        weights = tf.exp(tf.convert_to_tensor(weights, dtype_hint=tf.float32))

        if not resample_num:
            resample_num = ps.shape(weights)[-1]

        points_shape = ps.concat([ps.shape(weights)[:-1],
                                  [resample_num]], axis=0)
        # Draw an offset for every element of an event, with size [b, Ns]
        interval_width = ps.cast(1. / resample_num, dtype=weights.dtype)
        offsets = uniform.Uniform(low=ps.cast(0., dtype=weights.dtype),
                                  high=interval_width).sample(
            points_shape, seed=seed)

        # The unit interval is divided into equal partitions and each point
        # is a random offset into a partition.
        resampling_space = tf.linspace(
            start=ps.cast(0., dtype=weights.dtype),
            stop=1 - interval_width,
            num=resample_num) + offsets

        # Resampling
        # TODO: would weights would be multiple dimensions?
        cdf_weights = tf.concat([tf.math.cumsum(weights, axis=-1)[..., :-1],
                               tf.ones([1,], dtype=weights.dtype)],
                               axis=-1)

        resample_index = tf.searchsorted(cdf_weights, resampling_space)

        return resample_index


def _resample_systematic(weights, resample_num, seed, name=None):
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
        weights = tf.exp(tf.convert_to_tensor(weights, dtype_hint=tf.float32))

        if not resample_num:
            resample_num = ps.shape(weights)[-1]

        # Draw an offset for whole events, with size [b, ]
        interval_width = ps.cast(1. / resample_num, dtype=weights.dtype)
        offsets = uniform.Uniform(low=ps.cast(0., dtype=weights.dtype),
                                  high=interval_width).sample(seed=seed)

        # The unit interval is divided into equal partitions and each point
        # is a random offset into a partition.
        resampling_space = tf.linspace(
            start=ps.cast(0., dtype=weights.dtype),
            stop=1 - interval_width,
            num=resample_num) + offsets

        # Resampling

        cdf_weights = tf.concat([tf.math.cumsum(weights, axis=-1)[..., :-1],
                               tf.ones([1,], dtype=weights.dtype)],
                               axis=-1)

        # already batch based method
        resample_index = tf.searchsorted(cdf_weights, resampling_space)

        return resample_index


def _resample_multinomial(weights, resample_num, seed, name=None):
    """ This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the
    resampled point based on a uniformly distributed random number. Run time
    is O(n log n). You do not want to use this algorithm in practice; for some
    reason it is popular in blogs and online courses so I included it for
    reference.

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
    with tf.name_scope(name or 'resample_multinomial'):
        weights = tf.exp(tf.convert_to_tensor(weights, dtype_hint=tf.float32))

        if not resample_num:
            resample_num = ps.shape(weights)[-1]

        searchpoints = uniform.Uniform(low=ps.cast(0., dtype=weights.dtype),
                                  high=ps.cast(1., dtype=weights.dtype)).sample(resample_num, seed=seed)

        cdf_weights = tf.concat([tf.math.cumsum(weights, axis=-1)[..., :-1],
                               tf.ones([1,], dtype=weights.dtype)],
                               axis=-1)

        resample_index = tf.searchsorted(cdf_weights, searchpoints)

        # resample_index = Multinomial(probs=weights, total_count=ps.cast(resample_num, dtype=weights.dtype)).sample(1, seed=seed)

        return tf.sort(resample_index)