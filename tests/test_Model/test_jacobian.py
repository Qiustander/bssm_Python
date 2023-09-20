import pytest
import tensorflow as tf
import numpy as np
from Models.nonlinear_function_type import jacobian_fn


class TestJacobian:
    dtype = tf.float32

    @pytest.mark.parametrize(("dtype"),
                             [(dtype)])
    def test_gradient_no_batch(self, dtype):
        transition_fn = lambda t, x: tf.cast(tf.stack([tf.sin(x[..., 0])], axis=-1), dtype=dtype)

        transition_fn_grad = jacobian_fn(transition_fn)

        x = tf.random.normal([1,])

        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_equal(transition_fn_grad(random_idx, x), tf.cos(x))
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (1, 1))])
        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_equal(transition_fn_grad(random_idx, x), tf.cos(x))
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (1, 1))])

    @pytest.mark.parametrize(("dtype"),
                             [(dtype)])
    def test_gradient_no_batch_multivariate(self, dtype):

        transition_fn = lambda t, x: tf.cast(tf.sin(x), dtype=dtype)
        transition_fn_grad = jacobian_fn(transition_fn)
        dim = 3

        x = tf.random.normal([dim,])
        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_equal(transition_fn_grad(random_idx, x), tf.linalg.diag(tf.cos(x)))
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (dim, dim))])
        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_equal(transition_fn_grad(random_idx, x), tf.linalg.diag(tf.cos(x)))
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (dim, dim))])

    @pytest.mark.parametrize(("dtype"),
                             [(dtype)])
    def test_gradient_batch(self, dtype):

        transition_fn = lambda t, x: tf.cast(tf.stack([tf.sin(x[..., 0])], axis=-1), dtype=dtype)
        transition_fn_grad = jacobian_fn(transition_fn)

        x = tf.random.normal([100, 1])
        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (100, 1, 1))])
        tf.debugging.assert_equal(transition_fn_grad(random_idx, x), tf.cos(x)[..., tf.newaxis])
        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_equal(transition_fn_grad(random_idx, x), tf.cos(x)[..., tf.newaxis])
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (100, 1, 1))])


    @pytest.mark.parametrize(("dtype"),
                             [(dtype)])
    def test_gradient_batch_mv(self, dtype):

        transition_fn = lambda t, x: tf.cast(tf.sin(x), dtype=dtype)
        transition_fn_grad = jacobian_fn(transition_fn)

        dim = 4
        x = tf.random.normal([100, dim])
        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (100, dim, dim))])
        tf.debugging.assert_equal(transition_fn_grad(random_idx, x), tf.linalg.diag(tf.cos(x)))
        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_equal(transition_fn_grad(random_idx, x), tf.linalg.diag(tf.cos(x)))
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (100, dim, dim))])


    @pytest.mark.parametrize(("dtype"),
                             [(dtype)])
    def test_gradient_mv_non_square(self, dtype):

        # state dim = 4, obs dim = 3
        transition_fn = lambda t, x: tf.cast(tf.stack([x[..., 0] ** 2, x[..., 1] ** 3,
                                                        0.5 * x[..., 2] + 2 * x[..., 3] + x[..., 0] + x[..., 1]],
                                                       axis=-1), dtype=dtype)

        transition_fn_grad = jacobian_fn(transition_fn)
        cal_jacobian = lambda t, x: tf.cast(tf.stack(
            ([2*x[..., 0], 0., 0., 0.],
            [0., 3*x[..., 1] ** 2, 0., 0.],
            [1., 1., 0.5, 2.]),axis=0), dtype=dtype)

        dim = 4
        x = tf.random.normal([100, dim])
        random_idx = tf.experimental.numpy.random.randint(low=0, high=999)
        tf.debugging.assert_shapes([(transition_fn_grad(random_idx, x), (100, 3, dim))])
        tfp_jacobian = transition_fn_grad(random_idx, x)
        for i in range(x.shape[0]):
            tf.debugging.assert_equal(tfp_jacobian[i], cal_jacobian(random_idx, x[i]))
