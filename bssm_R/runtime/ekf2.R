library("bssm")
n <- 150
x <- y <- numeric(n)
x[0] <- 0.1
y[1] <- rnorm(1, exp(x[1]), 0.1)
for(i in 1:(n-1)) {
 x[i+1] <- rnorm(1, sin(x[i]), 0.1)
 y[i+1] <- rnorm(1, exp(x[i+1]), 0.1)
}

pntrs <- cpp_example_model("nlg_sin_exp")

model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1,
  Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn,
  Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
  theta = c(log_H = log(0.1), log_R = log(0.1)),
  log_prior_pdf = pntrs$log_prior_pdf,
  n_states = 1, n_etas = 1, state_names = "state")

total_time <- 0
for (x in 1:20) {
    start_time <- Sys.time()
  out_ekf <- ekf(model_nlg, iekf_iter = 0)
    end_time <- Sys.time()
    total_time <- total_time + end_time - start_time
}
print(total_time/20)