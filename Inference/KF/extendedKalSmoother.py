import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


# @tf.function
def extended_kalman_smoother(ssm_model, observations):
    """ Conduct the extended Kalman Smoother
    Args:
        ssm_model: model object (nonlinear model)
        observations: observed time series
    Returns:
        infer_result of EKF
    """
    (filtered_means, filtered_covs,
    predicted_means, predicted_covs, _, _,_,_) = tfp.experimental.sequential.extended_kalman_filter(
            observations,
            ssm_model.initial_state_prior,
            ssm_model.transition_plusnoise_fn,
            ssm_model.observation_plusnoise_fn,
            ssm_model.transition_fn_grad,
            ssm_model.observation_fn_grad,
            name=None
        )
    smoothed_means, smoothed_covs = backward_smoothing_pass(
        filtered_means, filtered_covs,
        predicted_means, predicted_covs)


def backward_smoothing_pass(filtered_means, filtered_covs,
        predicted_means, predicted_covs):

    return (smoothed_means, smoothed_covs)



