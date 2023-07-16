import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions
import time

sample_point = 4000

big_mean = tf.stack([tf.linspace(0.4, 0.8, sample_point),
                     .2*tf.linspace(0.4, 0.8, sample_point),
                     0.3*tf.linspace(0.4, 0.8, sample_point)],  axis=-1)

x_batch = tfd.MultivariateNormalLinearOperator(loc=big_mean,
                                scale=tf.linalg.LinearOperatorFullMatrix(tf.linalg.diag([0.1, 0.2, 0.3]))) # batch 4000, dim 3 for a sample

x_seq = lambda x: tfd.MultivariateNormalLinearOperator(loc=x,
                                scale=tf.linalg.LinearOperatorFullMatrix(tf.linalg.diag([0.1, 0.2, 0.3]))) # dim 3 for a sample

start_time = time.time()
one_sample  = x_batch.sample()
end_time = time.time()
print(f"batch sampling: {end_time-start_time}")
print(one_sample.shape)


start_time = time.time()
one_sample = tf.vectorized_map(lambda x: x_seq(x).sample(), big_mean)
end_time = time.time()
print(f"sequential sampling: {end_time-start_time}")
print(one_sample.shape)