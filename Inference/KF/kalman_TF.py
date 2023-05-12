# Python specific imports that will make our job easier and our code prettier
from collections import namedtuple
from functools import partial
import math
import time
from tqdm.auto import trange, tqdm

# TF specific imports that we will use to code the logic
from tensorflow import function
import tensorflow as tf
import tensorflow_probability as tfp

# Auxiliary libraries that we will use to report results and create the data
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

mm = tf.linalg.matmul
mv = tf.linalg.matvec

"""
Implement the Kalman filter via Tensorflow 
"""

class KalmanFilter:
    """
    Implement the Kalman filter via Tensorflow Probability official model

    ssm_ulg & ssm_mlg: tfd.LinearGaussianStateSpaceModel
    bsm_lg: tfp.sts.LocalLevelStateSpaceModel

    """
    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        def_kfmethod = getattr(self, f"kf_{kwargs['model_type']}")

        self.infer_result = def_kfmethod(model=kwargs['model'])

    def kf_linear_gaussian(self, model):
        """Kalman Filter for Linear Gaussian case.
        Args:
            model: bssm model object
            The input and output control variables are combined in the noise process.
        Returns:
            log_likelihoods:    Per-timestep log marginal likelihoods
                                log p(x[t] | x[:t-1]) evaluated at the input x, as a Tensor of
                                shape sample_shape(x) + batch_shape + [num_timesteps].
                                If final_step_only is True, this will instead be the cumulative
                                log marginal likelihood at the final step.
            filtered_means:	Means of the per-timestep filtered marginal
                            distributions p(z[t] | x[:t]), as a Tensor of shape
                            sample_shape(x) + batch_shape + [num_timesteps, latent_size].
            filtered_covs:	Covariances of the per-timestep filtered marginal
                            distributions p(z[t] | x[:t]), as a Tensor of shape sample_shape(x)
                             + batch_shape + [num_timesteps, latent_size,latent_size].
                             Since posterior covariances do not depend on observed data,
                             some implementations may return a Tensor whose shape omits
                             the initial sample_shape(x).
            predicted_means:	Means of the per-timestep predictive
                            distributions over latent states, p(z[t+1] | x[:t]),
                            as a Tensor of shape sample_shape(x) + batch_shape +
                            [num_timesteps, latent_size].
            predicted_covs:	Covariances of the per-timestep predictive
                        distributions over latent states, p(z[t+1] | x[:t]),
                        as a Tensor of shape sample_shape(x) + batch_shape +
                        [num_timesteps, latent_size, latent_size]. Since posterior
                        covariances do not depend on observed data, some implementations
                        may return a Tensor whose shape omits the initial sample_shape(x).
            observation_means:	Means of the per-timestep predictive
                        distributions over observations, p(x[t] | x[:t-1]),
                        as a Tensor of shape sample_shape(x) + batch_shape +
                        [num_timesteps, observation_size].
            observation_covs:	Covariances of the per-timestep predictive
                        distributions over observations, p(x[t] | x[:t-1]), as a Tensor
                        of shape sample_shape(x) + batch_shape +
                        [num_timesteps,observation_size, observation_size].
                        Since posterior covariances do not depend on observed data,
                        some implementations may return a Tensor whose shape omits the
                        initial sample_shape(x).

        """

        observation = model.y
        with tf.device('/CPU:0'):
             (fms, fPs) = kfilter(model, observation)
        return (fms, fPs)


@partial(tf.function, experimental_relax_shapes=True)
def kfilter(model, observations):
    def body(mean_cov, y):
        mu, Pt = mean_cov
        mu = mv(model.state_mtx, mu)
        Pt = model.state_mtx @ mm(Pt, model.state_mtx, transpose_b=True) + model.Q

        S = model.obs_mtx @ mm(Pt, model.obs_mtx, transpose_b=True) + model.R

        chol = tf.linalg.cholesky(S)
        Kt = tf.linalg.cholesky_solve(chol, model.obs_mtx @ Pt)

        mu = mu + mv(Kt, y - mv(model.obs_mtx, mu), transpose_a=True)
        Pt = Pt - mm(Kt, S, transpose_a=True) @ Kt
        return mu, Pt

    fms, fPs = tf.scan(body, observations, (model.prior_mean, model.prior_cov))
    return fms, fPs