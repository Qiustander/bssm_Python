import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.ssm_nlg import NonlinearSSM
from Models.check_argument import *
import os.path as pth
import os

numpy2ri.activate()

# Must call the function via the package, if the desired function is
# linked with other functions in the package. The simple call via ro.r("""source""")
# will only create a simple object that miss the link.
base = importr('base', lib_loc="/usr/lib/R/library")
bssm = importr('bssm', lib_loc=f"{pth.expanduser('~')}/R/x86_64-pc-linux-gnu-library/4.3")
stat = importr('stats', lib_loc="/usr/lib/R/library")

ro.r("""
n <- 150
x <- y <- numeric(n)
x[0] <- 0.1
y[1] <- rnorm(1, exp(x[1]), 0.1)
for(i in 1:(n-1)) {
 x[i+1] <- rnorm(1, sin(x[i]), 0.1)
 y[i+1] <- rnorm(1, exp(x[i+1]), 0.2)
}

pntrs <- cpp_example_model("nlg_sin_exp")

model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1,
  Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn,
  Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
  theta = c(log_H = log(0.1), log_R = log(0.1)),
  log_prior_pdf = pntrs$log_prior_pdf,
  n_states = 1, n_etas = 1, state_names = "state")

    """)

observation = np.array(ro.r("y"))

size_y, observation = check_y(observation.astype("float32"))  # return time length and feature numbers
num_timesteps, observation_size = size_y

model_obj = NonlinearSSM.create_dist(num_timesteps=num_timesteps,
                         observation_size=observation_size,
                         latent_size=1,
                         initial_state_mean=0.1,
                         initial_state_cov=0.,
                         state_noise_std=0.1,
                         obs_noise_std=0.2,
                         nonlinear_type="nlg_sin_exp")