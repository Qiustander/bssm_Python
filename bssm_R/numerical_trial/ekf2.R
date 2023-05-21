library("bssm")
n <- 50
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

out_ekf <- ekf(model_nlg, iekf_iter = 0)
out_iekf <- ekf(model_nlg, iekf_iter = 100)
ts.plot(cbind(x, out_ekf$att, out_iekf$att), col = c("red", "green", "blue"), lwd=2)
legend("bottomright", legend = c("x", "ekf_x", "iekf_x"), col = 1:3, lty = 1)
