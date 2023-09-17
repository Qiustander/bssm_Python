import types
from Synthetic_Data.stochastic_volatility import gen_ssm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Inference.MCMC.particle_marginal_mh import particle_marginal_metropolis_hastings
from Inference.SMC.bootstrap_particle_filter import bootstrap_particle_filter, particle_filter
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import arviz as az
from tensorflow_probability.python.internal import prefer_static as ps
import time
import logging
from Models.nonlinear_function_type import _batch_multiply, _process_mtx_tv

logging.basicConfig(level=logging.INFO)

tfd = tfp.distributions
tfb = tfp.bijectors

"""
For 1D Gaussian noise case, parameters theta: sigma_x, sigma_y
"""

VALIDATE_ARGS = True
num_timesteps = 300
num_particles = 600
num_samples = 3000
particle_method = 'bsf'
state_dim = 3
observation_dim = state_dim
"""
Generate data 
"""
true_mu = tf.random.normal(shape=[state_dim], mean=0.0, stddev=2.0)
def_coefficient = tf.random.uniform(shape=[state_dim], minval=-0.9, maxval=0.98)
true_rho = tf.linalg.diag(def_coefficient)
A = 0.5 * np.random.rand(state_dim, state_dim)
noise_cov = tf.convert_to_tensor(np.eye(state_dim) * 0.5 + A.dot(A), dtype=true_mu.dtype)
true_noise = tf.linalg.cholesky(noise_cov)
print(f'true rho: {true_rho}')
print(f'true mu: {true_mu}')
print(f'true noise: {true_noise}')

ssm_model = gen_ssm(num_timesteps=num_timesteps,
                    state_dim=state_dim,
                    state_mtx_noise=true_noise,
                    rho=true_rho,
                    mu=true_mu,
                    observed_dim=observation_dim,
                    default_mode='multivariate')

true_state, observations = ssm_model.simulate(num_timesteps)
# true_state = true_state[-num_timesteps:]
# observations = observations[-num_timesteps:]

# prior specification
PRIOR_MEAN_MU = tf.constant([0.] * state_dim)
PRIOR_STD_MU = tf.linalg.diag(tf.constant([2.0] * state_dim))
PRIOR_DF = state_dim + 2
PRIOR_SCALE = tf.linalg.diag(tf.random.uniform([state_dim]) + 0.2) / PRIOR_DF
N_CHAINS = 12

sample_mean = tf.reduce_mean(true_state, axis=0)
sample_std = tf.linalg.cholesky(tfp.stats.covariance(true_state, sample_axis=0))
print(f'sample state mean: {sample_mean}')
print(f'sample state std: {sample_std}')

replicate_observations = tf.tile(tf.expand_dims(observations, axis=1), multiples=[1, N_CHAINS, 1])

"""
Posterior
"""
one_minus_rho = tf.eye(true_rho.shape[-1]) - true_rho
last_obs = tf.einsum('ij, bj -> bi', true_rho, true_state[:-1] - true_mu)
prior_cov = tf.matmul(PRIOR_STD_MU, PRIOR_STD_MU, transpose_b=True)
coeff_mu = tf.linalg.inv(prior_cov) + tf.matmul(tf.linalg.inv(noise_cov),
                                                tf.matmul(one_minus_rho, one_minus_rho, transpose_b=True)) * (
               true_state.shape[0])
last_obs2 = tf.einsum('ij, bj -> bi', true_rho, true_state[:-1])
coeff = tf.linalg.matvec(one_minus_rho, tf.reduce_sum(true_state[1:] - last_obs2, axis=0))
sum_xt = tf.matmul(true_state[:-1] - true_mu, true_state[1:] - true_mu, transpose_a=True)
sum_xt2 = tf.matmul(true_state[:-1] - true_mu, true_state[:-1] - true_mu, transpose_a=True)

true_posterior_mean_mu = tf.linalg.matvec(tf.linalg.inv(coeff_mu),
                                          tf.linalg.matvec(tf.linalg.inv(noise_cov), coeff) +
                                          tf.linalg.matvec(tf.linalg.inv(prior_cov),
                                                           PRIOR_MEAN_MU))
print(f"posterior mu: {true_posterior_mean_mu}")

true_posterior_mean_rho = tf.linalg.diag_part(tf.matmul(tf.linalg.inv(sum_xt2), sum_xt))
print(f"posterior rho: {true_posterior_mean_rho}")

temp = true_state[1:] - true_mu - last_obs
posterior_a = int(PRIOR_DF + ps.shape(true_state)[0] - 1)
posterior_b = tf.linalg.inv(tf.linalg.inv(PRIOR_SCALE) + tf.matmul(temp, temp, transpose_a=True))

poster_dist = tfd.WishartTriL(
    df=posterior_a,
    scale_tril=tf.linalg.cholesky(posterior_b))
precision = poster_dist.mean()
true_posterior_noise_mean = tf.linalg.inv(tf.linalg.cholesky(precision))
true_posterior_noise_std = poster_dist.stddev()


# print(f"posterior noise std: {true_posterior_noise_mean}")


class NewClassInstance:

    def __init__(self, original_instance):
        for attr_name, attr_value in original_instance.__dict__.items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in original_instance.__class__.__dict__.items():
            if isinstance(attr_value, property):
                setattr(NewClassInstance, attr_name, property(attr_value.fget, attr_value.fset, attr_value.fdel))

        # Currently, we could not trace the SMC results in the log_likelihood because of the wrapped transformed transition kernel.


operator_eye_mtx_big = tf.linalg.LinearOperatorDiag(tf.ones([N_CHAINS, state_dim ** 2]) + 1e-5)


class StochasticVolativitySSM(NewClassInstance):
    """Define the log_prior
    """

    """Define update function for ssm model
    """

    def update_model(self, sigma, mu, rho):
        new_class = StochasticVolativitySSM(self)

        sigma = sigma
        prior_mean = mu
        noise_cov = tf.matmul(sigma, sigma, transpose_b=True)

        # stationary mean
        operator_rho = tf.linalg.LinearOperatorDiag(rho)
        kron_operator = tf.linalg.LinearOperatorKronecker([operator_rho, operator_rho])
        subtract_kron_operator = tf.linalg.LinearOperatorFullMatrix(operator_eye_mtx_big.to_dense() -
                                                                    kron_operator.to_dense())
        updated_prior_cov = tf.einsum('bcd, bd -> bc',
                                      tf.linalg.LinearOperatorInversion(subtract_kron_operator).to_dense(),
                                      tf.reshape(noise_cov, [N_CHAINS, -1]))
        updated_prior_cov = tf.reshape(updated_prior_cov, [N_CHAINS, state_dim, state_dim])

        new_class._initial_state_prior = tfd.MultivariateNormalTriL(
            loc=prior_mean,
            scale_tril=tf.linalg.cholesky(updated_prior_cov))

        mu_vector_fn = _process_mtx_tv(mu, 2)
        transition_matrix_fn = _process_mtx_tv(operator_rho.to_dense(), 3)
        transition_fn = _batch_multiply(transition_matrix_fn)

        new_class._transition_dist = lambda t, x: tfd.MultivariateNormalTriL(
            loc=mu_vector_fn(t) + transition_fn(t, (x - mu_vector_fn(t))),
            scale_tril=sigma)

        return new_class

    """Define data likelihood for ssm model
    """


def log_theta_prior():
    return tfd.JointDistributionNamed(dict(
        mu=tfd.MultivariateNormalTriL(loc=PRIOR_MEAN_MU,
                                      scale_tril=PRIOR_STD_MU),
        sigma=tfd.WishartTriL(
            df=PRIOR_DF,
            scale_tril=tf.linalg.cholesky(PRIOR_SCALE)),
        rho=tfd.Uniform(low=-0.99, high=0.99)))


def initial_theta():
    """
    Returns: initial precision matrix estimation
    """

    return [tf.linalg.diag(tf.random.uniform([N_CHAINS, state_dim]) + 0.2),  # sigma
            tf.random.normal([N_CHAINS, state_dim]),  # mu
            tf.random.uniform([N_CHAINS, state_dim])]  # rho


def _smc_trace_fn(state, kernel_results):
    return (state.particles,
            state.log_weights,
            kernel_results.accumulated_log_marginal_likelihood)


def log_target_dist(ssm_models, observations, num_particles):
    def _log_likelihood(prec, mu, rho):
        # update model

        precisions_cholesky = tf.linalg.cholesky(prec)
        covariances = tf.linalg.cholesky_solve(
            precisions_cholesky, tf.linalg.eye(state_dim, batch_shape=[N_CHAINS]))
        sigma = tf.linalg.cholesky(covariances)

        new_class = ssm_models.update_model(sigma=sigma, mu=mu, rho=rho)

        # Conduct SMC
        def _run_smc(initial_state_prior, transition_dist, observation_dist):
            result = particle_filter(
                observations=observations,
                initial_state_prior=initial_state_prior,
                transition_fn=transition_dist,
                observation_fn=observation_dist,
                num_particles=num_particles,
                trace_fn=_smc_trace_fn,
                trace_criterion_fn=lambda *_: True)
            return result

        traced_results = _run_smc(initial_state_prior=new_class.initial_state_prior,
                                  transition_dist=new_class.transition_dist,
                                  observation_dist=new_class.observation_dist)

        return traced_results[-1][-1] \
            + log_theta_prior().log_prob({'sigma': sigma,
                                          'mu': mu,
                                          'rho': tf.zeros([N_CHAINS, ])})

    return _log_likelihood


sv_ssm = StochasticVolativitySSM(ssm_model)


@tf.function(jit_compile=True)
def run_mcmc():
    transformed_bijector = tfb.Chain([
        # step 3: take the product of Cholesky factors
        tfb.CholeskyOuterProduct(),
        # step 2: exponentiate the diagonals
        tfb.TransformDiagonal(tfb.Exp()),
        # step 1: map a vector to a lower triangular matrix
        tfb.FillTriangular(),
    ])

    unconstrained_to_precision = tfb.JointMap(
        bijectors=[transformed_bijector, tfb.Identity(), tfb.Tanh()]
    )

    result = particle_marginal_metropolis_hastings(num_samples,
                                                   target_dist=log_target_dist(sv_ssm, replicate_observations,
                                                                               num_particles),
                                                   transformed_bijector=unconstrained_to_precision,
                                                   init_state=initial_theta(),
                                                   num_burnin_steps=int(3 * num_samples),
                                                   num_steps_between_results=0,
                                                   seed=None,
                                                   name=None)
    return result


if __name__ == '__main__':

    start = time.time()
    mcmc_result = run_mcmc()
    end = time.time()
    print(f" execute time {end - start}")
    mcmc_trace = mcmc_result.trace_results
    mcmc_state = mcmc_result.states
    if isinstance(mcmc_state, list):
        mcmc_state = [mcmc_state[i].numpy() for i in range(len(mcmc_state))]
    else:
        mcmc_state = [mcmc_state]
    parameter_names = ['noise_std', 'mean', 'coefficients']
    posterior_trace = az.from_dict(
        posterior={
            k: np.swapaxes(v, 0, 1) for k, v in zip(parameter_names, mcmc_state)
        },
        # sample_stats={k: np.swapaxes(v, 0, 1) for k, v in mcmc_result.items()},
        observed_data={"observations": true_state.numpy()},
        coords={"feature_dim": np.arange(state_dim), "datalen": np.arange(true_state.shape[0])},
        dims={"observations": ["datalen", "feature_dim"]},
    )
    az.summary(posterior_trace)
    az.plot_trace(posterior_trace)
    az.plot_rank(posterior_trace)
    plt.show()

xxx = 1
