import matplotlib.pyplot as plt
import tensorflow as tf
import arviz as az
import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import distribution_util as dist_util

def run_pmcmc_diagnostic(mcmc_states,
                         mcmc_traces,
                         observations,
                         ssm_model,
                         sample_sizes,
                         parameter_names):
    """ Run PMCMC Diagnostics
    Args:
        ssm_model:
        mcmc_states:
        mcmc_traces:

    Returns:

    """

    @tf.function(autograph=True)
    def gen_posterior_samples(sample_size):
        posterior_predictive_observation = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        sigma_all = mcmc_states[..., 0:1]
        mu_all = mcmc_states[..., 1:2]
        rho_all = mcmc_states[..., 2:3]
        # sigma_all, mu_all, rho_all = mcmc_states
        overall_mcmc_samples = sigma_all.shape[0]
        overall_chains = sigma_all.shape[1]
        if sample_size > overall_mcmc_samples:
            raise ValueError("Not enough MCMC samples")

        for i in range(sample_size):
            choose_chain = tf.random.uniform(shape=(), minval=0, maxval=overall_chains - 1, dtype=tf.int32)
            choose_sample = tf.random.uniform(shape=(), minval=int(overall_mcmc_samples/1.5),
                                              maxval=overall_mcmc_samples - 1, dtype=tf.int32)
            sigma = sigma_all[choose_sample, choose_chain]
            mu = mu_all[choose_sample, choose_chain]
            rho = rho_all[choose_sample, choose_chain]
            # chain , shape

            new_svm_model = ssm_model.update_model(sigma=sigma,
                                            mu=mu,
                                            rho=rho)

            simulated_observations, _ = new_svm_model.simulate()

            posterior_predictive_observation = posterior_predictive_observation.write(i, simulated_observations)

        return posterior_predictive_observation.stack()

    if isinstance(mcmc_states, list):
        mcmc_states = [mcmc_states[i].numpy() for i in range(len(mcmc_states))]

    np.savez('mcmc_states.npy', mcmc_states, observations)

    predictive_observation = gen_posterior_samples(sample_sizes)
    if not isinstance(mcmc_states, list):
        mcmc_states = [mcmc_states[..., i:i+1] for i in range(mcmc_states.shape[-1])]
    posterior_trace = az.from_dict(
        posterior={
            k: np.swapaxes(v, 0, 1) for k, v in zip(parameter_names, mcmc_states)
        },
        posterior_predictive={"observations": predictive_observation[tf.newaxis, ...]},
        observed_data={"observations": observations},
        coords={"feature": np.arange(observations.shape[-1]),
                "obs_size":np.arange(observations.shape[0])},
        dims={"observations": ["obs_size", "feature"]},
    )
    print(az.summary(posterior_trace))
    az.plot_ppc(posterior_trace, group="posterior", num_pp_samples=sample_sizes)
    az.plot_trace(posterior_trace)
    az.plot_rank(posterior_trace)
    plt.show()

    r_hat_mcmc = tfp.mcmc.potential_scale_reduction(
        mcmc_states,
        independent_chain_ndims=1,
        split_chains=True)
    print(f"R^hat for MCMC algorithm {r_hat_mcmc}")

    ess_mcmc = tfp.mcmc.effective_sample_size(
        mcmc_states,
        cross_chain_dims=[1] * len(mcmc_states),
    )
    print(f"Effective sample size for MCMC algorithm {ess_mcmc}")
