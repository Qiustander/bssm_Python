import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc import resample_stratified, resample_independent, resample_systematic, resample_deterministic_minimum_error
from Inference.SMC import resampling as resample
from tensorflow_probability.python.internal import prefer_static as ps
import pytest
tfd = tfp.distributions
from filterpy_resampling import stratified_resample, systematic_resample, multinomial_resample, residual_resample


class TestResampling:
    num_particles = 100
    seed = 123

    @pytest.mark.parametrize(("num_particles", "seed"),
                             [(num_particles, seed)])
    def test_stratified_resample(self, num_particles, seed):

        # prob = tfd.uniform.Uniform(low=ps.cast(0., dtype=tf.float32),
        #                       high=ps.cast(0.7, dtype=tf.float32)).sample(
        #                           num_particles, seed=seed)
        tf.random.set_seed(seed)
        prob = tfd.Normal(loc=0.4, scale=0.4).sample(num_particles)
        log_prob = tf.nn.log_softmax(prob)

        tf.random.set_seed(seed)
        resample_indx_tfp = resample_stratified(log_prob, num_particles, (), seed=seed)

        tf.random.set_seed(seed)
        resample_indx_self = resample._resample_stratified(log_prob, resample_num=num_particles, seed=seed)

        tf.random.set_seed(seed)
        interval_width = ps.cast(1. / num_particles, dtype=tf.float32)
        offsets = tfd.uniform.Uniform(low=ps.cast(0., dtype=tf.float32),
                                  high=interval_width).sample(num_particles, seed=seed)
        # resample_indx_filterpy = stratified_resample(tf.exp(log_prob).numpy(), offsets)

        tf.debugging.assert_equal(resample_indx_tfp, resample_indx_self)

    @pytest.mark.parametrize(("num_particles", "seed"),
                             [(num_particles, seed)])
    def test_systematic_resample(self, num_particles, seed):

        tf.random.set_seed(seed)
        prob = tfd.Normal(loc=0.4, scale=0.4).sample(num_particles)
        log_prob = tf.nn.log_softmax(prob)

        tf.random.set_seed(seed)
        resample_indx_tfp = resample_systematic(log_prob, num_particles, (), seed=seed)
        tf.random.set_seed(seed)
        resample_indx_self = resample._resample_systematic(log_prob, resample_num=num_particles, seed=seed)
        tf.random.set_seed(seed)
        interval_width = ps.cast(1. / num_particles, dtype=tf.float32)
        offsets = tfd.uniform.Uniform(low=ps.cast(0., dtype=tf.float32),
                                  high=interval_width).sample(seed=seed)
        # resample_indx_filterpy = systematic_resample(tf.exp(log_prob).numpy(), offsets)

        tf.debugging.assert_equal(resample_indx_tfp, resample_indx_self)

    @pytest.mark.parametrize(("num_particles", "seed"),
                             [(num_particles, seed)])
    def test_multinomial_resample(self, num_particles, seed):
        tf.random.set_seed(seed)
        prob = tfd.Normal(loc=0.4, scale=0.4).sample(num_particles, seed=seed)
        log_prob = tf.nn.log_softmax(prob)

        tf.random.set_seed(seed)
        resample_indx_tfp = resample_independent(log_prob, num_particles, (), seed=seed)
        tf.random.set_seed(seed)
        resample_indx_self = resample._resample_multinomial(log_prob, resample_num=num_particles, seed=seed)
        # resample_indx_filterpy = multinomial_resample(tf.exp(log_prob).numpy())

        tf.debugging.assert_equal(resample_indx_tfp, resample_indx_self)

    @pytest.mark.parametrize(("num_particles", "seed"),
                             [(num_particles, seed)])
    def test_residual_resample(self, num_particles, seed):
        tf.random.set_seed(seed)
        prob = tfd.Normal(loc=0.4, scale=0.4).sample(num_particles, seed=seed)
        log_prob = tf.nn.log_softmax(prob)

        # tf.random.set_seed(seed)
        # resample_indx_tfp = tf.sort(resample_deterministic_minimum_error(log_prob, num_particles, (), seed=seed))
        tf.random.set_seed(seed)
        resample_indx_self = resample._resample_residual(log_prob, resample_num=num_particles, seed=seed)
        tf.random.set_seed(seed)
        resample_indx_filterpy = np.sort(residual_resample(tf.exp(log_prob).numpy(), seed=seed))

        tf.debugging.assert_equal(resample_indx_self, resample_indx_filterpy)

    @pytest.mark.parametrize(("num_particles", "seed"),
                             [(num_particles, seed)])
    def test_unnormalized_weights(self, num_particles, seed):

        tf.random.set_seed(seed)
        prob = tfd.Normal(loc=0.4, scale=0.4).sample(num_particles)
        normalized_log_prob = tf.nn.log_softmax(prob)
        log_prob = normalized_log_prob+1.2

        tf.random.set_seed(seed)
        resample_indx_tfp = resample_stratified(normalized_log_prob, num_particles, (), seed=seed)

        tf.random.set_seed(seed)
        resample_indx_self = resample._resample_stratified(log_prob, resample_num=num_particles, seed=seed)

        tf.debugging.assert_equal(resample_indx_tfp, resample_indx_self)
