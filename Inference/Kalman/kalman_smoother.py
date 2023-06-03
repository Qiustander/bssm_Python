import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import collections
from runtime_wrap import get_runtime

"""
The Kalman smoother is implemented based on the posterior_marginals function in Tensorflow
Probability Linear Gaussian SSM Model. The LinearGaussian SSM is implemented in Models.ssm_lg.
So this function is deprecated.
"""


class KalmanSmoother:
    """
    Implement the Kalman Smoother via Tensorflow Probability official model

    ssm_ulg & ssm_mlg: tfd.LinearGaussianStateSpaceModel
    bsm_lg: tfp.sts.LocalLevelStateSpaceModel

    """
    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        def_kfmethod = getattr(self, f"ksmoother_{kwargs['model_type']}")

        self.infer_result = def_kfmethod(model=kwargs['model'])


    @tf.function
    def ksmoother_linear_gaussian(self, model):
        """Kalman Smoother for Linear Gaussian case, including ssm_ulg & ssm_mlg & ar1_lg
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

        #TODO:  extend the innovation and observed noise to time-varying. Need modification in bssm_model
        observation = model.y
        time_len = observation.shape[0]

        # Must use LinearOperator for time-varying matrix since it provides batch shape.

        lg_model = tfp.distributions.LinearGaussianStateSpaceModel(
            num_timesteps=time_len,
            transition_matrix=tf.convert_to_tensor(model.state_mtx) if len(model.state_mtx.shape) == 2 else
                                process_tv(model, "state_mtx"), # Tensor or LinearOperator
            observation_matrix=tf.convert_to_tensor(model.obs_mtx) if len(model.obs_mtx.shape) == 2 else
                                process_tv(model, "obs_mtx"), # Tensor or LinearOperator.
            observation_noise=tfd.MultivariateNormalLinearOperator(
                                 loc=tf.convert_to_tensor(model.input_obs),
                            scale=tf.linalg.LinearOperatorFullMatrix(model.obs_mtx_noise)),
                                # tfd.MultivariateNormalLinearOperator
            transition_noise=tfd.MultivariateNormalLinearOperator(
                             loc=tf.convert_to_tensor(model.input_state),
                            scale= tf.linalg.LinearOperatorFullMatrix(model.state_mtx_noise)),
            initial_state_prior=tfd.MultivariateNormalLinearOperator(
                                     loc=model.prior_mean,
                                        scale= model.prior_cov if len(model.prior_cov.shape) == 1 else
                                        tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(model.prior_cov)))
        )

        infer_result = lg_model.posterior_marginals(tf.convert_to_tensor(observation))
        # infer_result = self.run_kf(lg_model, tf.convert_to_tensor(observation))
        return infer_result

    #
    # """Only for runtime testing
    # """
    # @staticmethod
    # @get_runtime(loop_time=10)
    # def run_kf(kf_model, observation):
    #     return kf_model.forward_filter(observation)


def process_tv(model, attritube):
    """
    Args:
        model: model object for time-varying process
        attritbue (str): matrix name of the model object: state_mtx/obs_mtx/state_mtx_noise/obs_mtx_noise
    Returns:
        matrix_tv: callable function that for t-th time point with wrapped matrix
    """
    def matrix_tv(t):
        return tf.linalg.LinearOperatorFullMatrix(tf.gather(tf.convert_to_tensor(getattr(model, attritube))
                                                  , indices=t, axis=-1))
    return matrix_tv





