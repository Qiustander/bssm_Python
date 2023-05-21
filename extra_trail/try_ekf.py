import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
from Models.check_argument import *

def transition_fn(x):
  return tfd.MultivariateNormalDiag(
      tf.stack(
          [x[..., 0] - 0.1 * x[..., 1]**3, x[..., 1]], axis=-1),
      scale_diag=[0.7, 0.2])

def transition_jacobian_fn(x):
  return tf.reshape(
    tf.stack(
        [tf.cast(1. - 0.1 * tf.cos(x[..., 0])**3, dtype=tf.float32), tf.cast(-0.3 * x[..., 1]**2, dtype=tf.float32),
        tf.zeros(x.shape[:-1], dtype=tf.float32), tf.ones(x.shape[:-1], dtype=tf.float32)], axis=-1),
    [2, 2])

observation_fn = lambda x: tfd.MultivariateNormalDiag(
    x[..., :1], scale_diag=[1.])
observation_jacobian_fn = lambda x: [[1., 0.]]

initial_state_prior = tfd.MultivariateNormalDiag(0., scale_diag=[1., 0.3])

prior_mean = tf.convert_to_tensor(check_prior_mean(0., 1), dtype=tf.float32)
prior_cov = tf.convert_to_tensor(check_prior_cov(1., 1), dtype=tf.float32)

initial_state_prior3 = tfd.MultivariateNormalLinearOperator(
    loc=prior_mean,
    scale=tf.linalg.LinearOperatorDiag(prior_cov) if tf.size(prior_cov) == 1 else
    tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(prior_cov)))

x = [tf.zeros((2,), dtype=tf.float32)]
for t in range(20):
  x.append(transition_fn(x[-1]).sample())
x = tf.stack(x)

observations=tf.cast(observation_fn(x).sample(), dtype=tf.float32)

# @tf.function
def wrap_ekf(*args):
    return tfp.experimental.sequential.extended_kalman_filter(*args)

g = wrap_ekf(
    observations,
    initial_state_prior,
    transition_fn,
    observation_fn,
    transition_jacobian_fn,
    observation_jacobian_fn)