import types
from Synthetic_Data.linear_gaussian import gen_data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Inference.MCMC.particle_marginal_mh import particle_marginal_metropolis_hastings
from Inference.SMC.bootstrap_particle_filter import bootstrap_particle_filter, particle_filter
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps

tfd = tfp.distributions
tfb = tfp.bijectors

"""
For 1D Gaussian noise case, parameters theta: sigma_x, sigma_y
"""
import logging

logging.basicConfig(level=logging.INFO)

VALIDATE_ARGS = True
num_timesteps = 100
num_particles = 300
num_samples = 4000
particle_method = 'bsf'
state_dim = 1
observation_dim = 1
testcase = 'univariate'

"""
Generate data 
"""
ssm_model = gen_data(testcase=testcase, num_timesteps=num_timesteps,
                     state_dim=state_dim, observed_dim=observation_dim)

true_state, observations = ssm_model.simulate()
PRIOR_DF = 3


class LinearGaussianSSM:
    """
    Precision Matrix Estimation - Inverse of the Covariance
    """

    def __init__(self, original_instance):
        for attr_name, attr_value in original_instance.__dict__.items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in original_instance.__class__.__dict__.items():
            if isinstance(attr_value, property):
                setattr(LinearGaussianSSM, attr_name, property(attr_value.fget, attr_value.fset, attr_value.fdel))

        # Currently, we could not trace the SMC results in the log_likelihood because of the wrapped transformed transition kernel.
    """Define the log_prior
    """

    def initial_theta(self):
        """
        Returns: initial precision matrix estimation
        """
        return [tf.constant([[0.5+np.random.uniform()]]), tf.constant([[0.5+np.random.uniform()]])]

    def log_theta_prior(self, sigma_x, sigma_y):
        sigma_x_prior = tfd.WishartTriL(
            df=PRIOR_DF,
            scale_tril=tf.constant([[1 / PRIOR_DF]]))
        sigma_y_prior = tfd.WishartTriL(
            df=PRIOR_DF,
            scale_tril=tf.constant([[1 / PRIOR_DF]]))
        return sigma_x_prior.log_prob(sigma_x) + sigma_y_prior.log_prob(sigma_y)

    """Define update function for ssm model
    """

    def update_model(self, sigma_x, sigma_y):
        # self._transition_noise_matrix = tf.convert_to_tensor(
        #     check_state_noise(sigma_x.numpy(), state_dim, num_timesteps), dtype=self.dtype)
        # self._observation_noise_matrix = tf.convert_to_tensor(
        #     check_obs_mtx_noise(sigma_y.numpy(), state_dim, num_timesteps), dtype=self.dtype)
        precision_x = tf.linalg.cholesky(sigma_x)
        precision_y = tf.linalg.cholesky(sigma_y)
        covariances_x = tf.linalg.cholesky_solve(
            precision_x, tf.linalg.eye(ps.shape(sigma_x)[-1]))
        covariances_y = tf.linalg.cholesky_solve(
            precision_y, tf.linalg.eye(ps.shape(sigma_y)[-1]))

        self._transition_noise_matrix = tf.linalg.cholesky(covariances_x)
        self._observation_noise_matrix = tf.linalg.cholesky(covariances_y)

        self._transition_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=self.transition_fn(t, x),
            scale_tril=self._transition_noise_matrix)
        self._observation_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=self.observation_fn(t, x),
            scale_tril=self._observation_noise_matrix
        )

    """Define data likelihood for ssm model
    """

    def log_target_dist(self, observations, num_particles):
        def _log_likelihood(sigma_x, sigma_y):
            # update model
            self.update_model(sigma_x, sigma_y)

            # Conduct SMC
            def _run_smc():
                def _smc_trace_fn(state, kernel_results):
                    return (state.particles,
                            state.log_weights,
                            kernel_results.accumulated_log_marginal_likelihood)

                result = particle_filter(
                    observations=observations,
                    initial_state_prior=self.initial_state_prior,
                    transition_fn=self.transition_dist,
                    observation_fn=self.observation_dist,
                    num_particles=num_particles,
                    trace_fn=_smc_trace_fn,
                    trace_criterion_fn=lambda *_: True)
                return result

            traced_results = _run_smc()
            #TODO: could not trace the results because of out of scope
            # self.smc_trace_results.append(traced_results)
            return traced_results[-1][-1] \
                   + self.log_theta_prior(sigma_x, sigma_y)

        return _log_likelihood


linear_gaussian_ssm = LinearGaussianSSM(ssm_model)


@tf.function
def run_mcmc():

    #tfb.Blockwise is opposite to tfb.Chain.
    # tfb.Chain executes the bijectors up-side down
    unconstrained_to_precision_chain = tfb.Chain([
        # step 3: take the product of Cholesky factors
        tfb.CholeskyOuterProduct(validate_args=VALIDATE_ARGS),
        # step 2: exponentiate the diagonals
        tfb.TransformDiagonal(tfb.Exp(validate_args=VALIDATE_ARGS)),
        # step 1: map a vector to a lower triangular matrix
        tfb.FillTriangular(validate_args=VALIDATE_ARGS),
    ])

    unconstrained_to_precision = tfb.JointMap(
        bijectors=[unconstrained_to_precision_chain, unconstrained_to_precision_chain]
    )

    result = particle_marginal_metropolis_hastings(linear_gaussian_ssm,
                                                   observations,
                                                   num_samples,
                                                   num_particles,
                                                   transformed_bijector=unconstrained_to_precision,
                                                   init_state=linear_gaussian_ssm.initial_theta(),
                                                   num_burnin_steps=int(num_samples // 2),
                                                   num_steps_between_results=0,
                                                   seed=None,
                                                   name=None)
    return result


@tf.function
def run_smc():
    result = bootstrap_particle_filter(ssm_model,
                                       observations,
                                       num_particles)
    return result


if __name__ == '__main__':
    mcmc_result = run_mcmc()
    smc_result = run_smc()

    # diagnostics
    plt.hist(mcmc_result.states[0])
    plt.show()
    plt.hist(mcmc_result.states[1])
    plt.show()

    mcmc_trace = mcmc_result.trace_results
