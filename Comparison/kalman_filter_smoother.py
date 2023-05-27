import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.bssm_model import *
from Inference.KF.kalman_smoother_TFP import KalmanSmoother as Ksmoother
from Inference.KF.kalman_TFP import KalmanFilter as KFTFP
import os.path as pth
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpu = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

# Automatic convertion between R and Python objects
numpy2ri.activate()

# Must call the function via the package, if the desired function is
# linked with other functions in the package. The simple call via ro.r("""source""")
# will only create a simple object that miss the link.
base = importr('base', lib_loc="/usr/lib/R/library")
bssm = importr('bssm', lib_loc=f"{pth.expanduser('~')}/R/x86_64-pc-linux-gnu-library/4.3")
stat = importr('stats', lib_loc="/usr/lib/R/library")


ro.r("""
n <- 150
x1 <- rnorm(n)
x2 <- rnorm(n)
b1 <- 1 + cumsum(rnorm(n, sd = 0.5))
b2 <- 2 + cumsum(rnorm(n, sd = 0.1))
y <- 1 + b1 * x1 + b2 * x2 + rnorm(n, sd = 0.1)

Z <- rbind(1, x1, x2)
H <- 0.1
T <- diag(3)
R <- diag(c(0, 1, 0.1))
a1 <- rep(0, 3)
P1 <- diag(10, 3)

# updates the model given the current values of the parameters
update_fn <- function(theta) {
  R <- diag(c(0, theta[1], theta[2]))
  dim(R) <- c(3, 3, 1)
  list(R = R, H = theta[3])
}
# prior for standard deviations as half-normal(1)
prior_fn <- function(theta) {
  if(any(theta < 0)) {
    log_p <- -Inf
  } else {
    log_p <- sum(dnorm(theta, 0, 1, log = TRUE))
  }
  log_p
}

model_r <- ssm_ulg(y, Z, H, T, R, a1, P1,
         init_theta = c(1, 0.1, 0.1),
         update_fn = update_fn, prior_fn = prior_fn,
         state_names = c("level", "b1", "b2"),
         # using default values, but being explicit for testing purposes
         C = matrix(0, 3, 1), D = numeric(1))
             infer_result <- kfilter(model_r)

""")
y = np.array(ro.r["y"])
obs_mtx = np.array(ro.r("Z"))
obs_mtx_noise = ro.r("H")
init_theta = ro.r("""c(1, 0.1, 0.1)""")
state_mtx = np.array(ro.r("T"))
state_mtx_noise = np.array(ro.r("R"))
prior_mean = np.array(ro.r("a1"))
prior_cov = np.array(ro.r("P1"))
input_state = ro.r("""matrix(0, 3, 1)""")
input_obs = np.array([0.])
r_result = ro.r["infer_result"]

model_obj = SSModel(model_name="ssm_ulg", y=y, state_dim=state_mtx.shape[0],
                    obs_mtx=obs_mtx, obs_mtx_noise=obs_mtx_noise,
                    init_theta=init_theta,
                    state_mtx=state_mtx, state_mtx_noise=state_mtx_noise,
                    prior_mean=prior_mean, prior_cov=prior_cov,
                  input_state=input_state, input_obs=input_obs,
                 )
infer_kf = KFTFP(model_type="linear_gaussian", model=model_obj).infer_result
infer_ksmoother = Ksmoother(model_type="linear_gaussian", model=model_obj).infer_result

true_state = np.array(ro.r("b1"))[..., None]
plt.plot(infer_kf.filtered_means.numpy()[:,1], color='blue', linewidth=1)
plt.plot(infer_ksmoother[0][:,1], color='green', linewidth=1)
plt.plot(true_state, '-.', color='red', linewidth=1)
plt.legend(("infer_kf", "infer_smoother", "true_state"))
plt.show()

print(f"MSE between filter and true state:{np.mean((infer_kf.filtered_means.numpy()[:,1] - true_state)**2)}")
print(f"MSE between smoother and true state:{np.mean((infer_ksmoother[0][:,1] - true_state)**2)}")

