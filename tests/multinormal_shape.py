import tensorflow_probability as tfp
import tensorflow as tf

tfd =tfp.distributions


# 10 indept chains, 3 parameters
mean = tf.random.normal([10,3])
std = tf.random.uniform([10, 3, 3])

dist = tfd.MultivariateNormalTriL(loc=mean,
                                  scale_tril=std)
print(dist) # batch 10, event 3


# 10 indept chains, 1 parameter, no shape
mean = tf.random.normal([10,])
std = tf.random.uniform([10, 10])

dist = tfd.MultivariateNormalTriL(loc=mean,
                                  scale_tril=std)
print(dist) # batch , event 10


# 10 indept chains, 1 parameter, [1,]
mean = tf.random.normal([10,1])
std = tf.random.uniform([10, 10])

dist = tfd.MultivariateNormalTriL(loc=mean,
                                  scale_tril=std)
print(dist) # batch 10, event 10, should need to tf.reshape to drop the last dimension

# 10 indept chains, 1 parameter, [1,]
mean = tf.random.normal([10, 1])
std = tf.random.uniform([10, 1, 1])

dist = tfd.MultivariateNormalTriL(loc=mean,
                                  scale_tril=std)
print(dist) # batch 10, event 1, correct


mean = tf.random.normal([3, 3])
std = tf.random.uniform([3, 1])

dist = tfd.MultivariateNormalTriL(loc=mean,
                                  scale_tril=std)

print(dist) # batch 3, event 3, wrong


mean = tf.random.normal([3, 3])
std = 0.2*tf.eye(3, batch_shape=[1])

dist = tfd.MultivariateNormalTriL(loc=mean,
                                  scale_tril=std)

print(dist) # batch 3, event 3, wrong, for matrix parameter, us original random_normal

