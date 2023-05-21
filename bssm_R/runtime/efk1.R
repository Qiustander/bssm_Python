library("bssm")
set.seed(1)
mu <- -0.2
rho <- 0.7
sigma_y <- 0.1
sigma_x <- 1
x <- numeric(150)
x[1] <- rnorm(1, mu, sigma_x / sqrt(1 - rho^2))
for(i in 2:length(x)) {
  x[i] <- rnorm(1, mu * (1 - rho) + rho * x[i - 1], sigma_x)
}
y <- rnorm(150, exp(x), sigma_y)

pntrs <- cpp_example_model("nlg_ar_exp")

model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1,
  Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn,
  Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
  theta = c(mu= mu, rho = rho,
    log_sigma_x = log(sigma_x), log_sigma_y = log(sigma_y)),
  log_prior_pdf = pntrs$log_prior_pdf,
  n_states = 1, n_etas = 1, state_names = "state")


total_time <- 0
for (x in 1:10) {
    start_time <- Sys.time()
    out_ekf <- ekf(model_nlg, iekf_iter = 0)
    end_time <- Sys.time()
    total_time <- total_time + end_time - start_time
}
print(total_time/10)
