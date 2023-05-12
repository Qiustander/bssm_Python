import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import os.path as pth
import os
import time
from Models.bssm_model import *
from Inference.KF.kalman_filter_R import KalmanFilter as KFR
from Inference.KF.kalman_TFP import KalmanFilter as KFTFP

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Automatic convertion between R and Python objects
numpy2ri.activate()

# Must call the function via the package, if the desired function is
# linked with other functions in the package. The simple call via ro.r("""source""")
# will only create a simple object that miss the link.
base = importr('base', lib_loc="/usr/lib/R/library")
bssm = importr('bssm', lib_loc=f"{pth.expanduser('~')}/R/x86_64-pc-linux-gnu-library/4.3")
stat = importr('stats', lib_loc="/usr/lib/R/library")

ro.r("""
    data("GlobalTemp", package = "KFAS")
    model_r <- ssm_mlg(GlobalTemp, H = matrix(c(0.15,0.05,0, 0.05), 2, 2),
      R = 0.05, Z = matrix(1, 2, 1), T = 1, P1 = 10,
      state_names = "temperature",
      # using default values, but being explicit for testing purposes
      D = matrix(0, 2, 1), C = matrix(0, 1, 1))
  infer_result <- kfilter(model_r)
""")

y = np.array(ro.r("model_r$y"))
obs_mtx = np.array(ro.r("model_r$Z"))
obs_mtx_noise = ro.r("model_r$H")
init_theta = ro.r("model_r$theta")
state_mtx = np.array(ro.r("model_r$T"))
state_mtx_noise = np.array(ro.r("model_r$R"))
prior_mean = np.array(ro.r("model_r$a1"))
prior_cov = np.array(ro.r("model_r$P1"))
input_state = np.array(ro.r("model_r$C"))
input_obs = np.array(ro.r("model_r$D"))
r_result = ro.r["infer_result"]


model_obj = SSModel(model_name="ssm_mlg", y=y, state_dim=state_mtx.shape[0],
                    obs_mtx=obs_mtx, obs_mtx_noise=obs_mtx_noise,
                    init_theta=init_theta,
                    state_mtx=state_mtx, state_mtx_noise=state_mtx_noise,
                    prior_mean=prior_mean, prior_cov=prior_cov,
                  input_state=input_state, input_obs=input_obs,
                 )

"""
TFP
"""
infer_result = KFTFP(model_type="linear_gaussian", model=model_obj).infer_result

"""
RPY2
"""
model_type_case = model_obj.model_type_case()
model_obj_R = model_obj._toRssmulg(prior_fn=ro.r("""prior_fn = function(theta) {0}"""),
                                         update_fn=ro.r("""update_fn = function(theta) {0}"""))
infer_result = KFR(model_type="linear_gaussian",
                            model=model_obj_R, model_type_case=model_type_case).infer_result


