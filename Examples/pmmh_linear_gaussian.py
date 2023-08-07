import types
from Models.ssm_nlg import NonlinearSSM
from Models.ssm_nlg import nonlinear_fucntion
from Synthetic_Data.linear_gaussian import gen_data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Inference.MCMC.particle_marginal_mh import particle_marginal_metropolis_hastings
from Inference.SMC.bootstrap_particle_filter import bootstrap_particle_filter, particle_filter
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.experimental.mcmc.particle_filter import particle_filter as pf

tfd = tfp.distributions

"""
For 1D Gaussian noise case, parameters theta: sigma_x, sigma_y
"""
import logging
logging.basicConfig(level=logging.INFO)

num_timesteps = 20
num_particles = 10
num_samples = 1000
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


class LinearGaussianSSM:

    def __init__(self, original_instance):
        for attr_name, attr_value in original_instance.__dict__.items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in original_instance.__class__.__dict__.items():
            if isinstance(attr_value, property):
                setattr(LinearGaussianSSM, attr_name, property(attr_value.fget, attr_value.fset, attr_value.fdel))

    """Define the log_prior
    """

    def initial_theta(self):
        return [0.1, 0.1]

    def log_theta_prior(self):
        # sigma_x_prior = tfd.WishartTriL(
        #               df=tf.constant(1.),
        #               scale_tril=tf.constant([[0.5]]))
        # sigma_y_prior = tfd.WishartTriL(
        #               df=tf.constant(1.),
        #               scale_tril=tf.constant([[0.5]]))
        sigma_x_prior = tfd.HalfNormal(2)
        sigma_y_prior = tfd.HalfNormal(2)

        def _calculate(sigma_x, sigma_y):

            return sigma_x_prior.log_prob(sigma_x) + sigma_y_prior.log_prob(sigma_y)

        return _calculate

    """Define update function for ssm model
    """

    def update_model(self, sigma_x, sigma_y):
        # self._transition_noise_matrix = tf.convert_to_tensor(
        #     check_state_noise(sigma_x.numpy(), state_dim, num_timesteps), dtype=self.dtype)
        # self._observation_noise_matrix = tf.convert_to_tensor(
        #     check_obs_mtx_noise(sigma_y.numpy(), state_dim, num_timesteps), dtype=self.dtype)
        self._transition_noise_matrix = sigma_x[tf.newaxis, tf.newaxis]
        self._observation_noise_matrix = sigma_y[tf.newaxis, tf.newaxis]

        self._transition_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
            loc=self.transition_fn(t,x),
            scale=tf.cond(tf.equal(tf.size(self._transition_noise_matrix), 1),
                          lambda: tf.linalg.LinearOperatorFullMatrix(tf.sqrt(self._transition_noise_matrix)),
                          lambda: tf.linalg.LinearOperatorFullMatrix(
                              tf.linalg.cholesky(self._transition_noise_matrix))))
        self._observation_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
            loc=self.observation_fn(t, x),
            scale=tf.cond(tf.equal(tf.size(self._observation_noise_matrix), 1),
                          lambda: tf.linalg.LinearOperatorFullMatrix(tf.sqrt(self._observation_noise_matrix)),
                          lambda: tf.linalg.LinearOperatorFullMatrix(tf.linalg.cholesky(self._observation_noise_matrix)))
        )

    """Define data likelihood for ssm model
    """

    def log_target_dist(self, observations):
        def _log_likelihood(sigma_x, sigma_y):

            # logging.info(f"sigma x {tf.get_static_value(sigma_x)}")
            # logging.info(f"sigma x {ps._get_static_value(sigma_x)}")
            # logging.info(f"sigma y {tf.get_static_value(sigma_y)}")
            # update model
            self.update_model(sigma_x, sigma_y)

            # Conduct SMC
            def _smc_trace_fn(state, kernel_results):
                return (state.particles,
                        state.log_weights,
                        kernel_results.accumulated_log_marginal_likelihood)

            (particles,  # num_time_step, particle_num, state_dim
             log_weights,
             accumulated_log_marginal_likelihood) = particle_filter(
                observations=observations,
                initial_state_prior=self.initial_state_prior,
                transition_fn=self.transition_dist,
                observation_fn=self.observation_dist,
                num_particles=num_particles,
                trace_fn=_smc_trace_fn,
                trace_criterion_fn=lambda *_: True)
            # import logging
            # logging.basicConfig(level=logging.DEBUG)
            # logging.info(f"sigma x {tf.get_static_value(accumulated_log_marginal_likelihood)}")
            # logging.info(f"sigma x {tf.get_static_value(particles)}")
            # logging.info(f"sigma x {tf.get_static_value(log_weights)}")

            return accumulated_log_marginal_likelihood[-1] \
                   + self.log_theta_prior()(sigma_x, sigma_y), [particles, log_weights]
        return _log_likelihood


linear_gaussian_ssm = LinearGaussianSSM(ssm_model)
linear_gaussian_ssm.log_target_dist = linear_gaussian_ssm.log_target_dist(observations)


@tf.function
def run_mcmc():
    result = particle_marginal_metropolis_hastings(linear_gaussian_ssm,
                                                   num_samples,
                                                   num_particles,
                                                   init_state=linear_gaussian_ssm.initial_theta(),
                                                   num_burnin_steps=int(num_samples // 3),
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
