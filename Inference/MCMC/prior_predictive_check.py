import matplotlib.pyplot as plt
import tensorflow as tf
import arviz as az
import numpy as np


def prior_predictive_check(ssm_model, observations, sample_sizes, prior_dist):
    """ Run prior predictive check with SSM.
    Args:
        ssm_model:

    Returns:

    """

    @tf.function(autograph=True)
    def gen_prior_samples(sample_size):
        prior_predictive_observation = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        for i in range(sample_size):
            gen_prior_parameters = prior_dist().sample()

            new_svm_model = ssm_model.update_model(sigma=gen_prior_parameters['sigma'],
                                            mu=gen_prior_parameters['mu'],
                                            rho=gen_prior_parameters['rho'][..., tf.newaxis])

            simulated_observations, _ = new_svm_model.simulate()

            prior_predictive_observation = prior_predictive_observation.write(i, simulated_observations)

        return prior_predictive_observation.stack()

    predictive_observation = gen_prior_samples(sample_sizes)

    prior_trace = az.from_dict(
        observed_data={"observations": observations},
        prior_predictive={"observations": predictive_observation[tf.newaxis, ...]},
        coords={"feature": np.arange(observations.shape[-1]),
                "obs_size":np.arange(observations.shape[0])},
        dims={"observations": ["obs_size", "feature"]},
    )

    az.plot_ppc(prior_trace, group="prior", num_pp_samples=sample_sizes)
    plt.show()
