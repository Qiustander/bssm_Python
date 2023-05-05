# non gaussian

data(poisson_series)
s <- sd(log(pmax(0.1, poisson_series)))
model <- bsm_ng(poisson_series, sd_level = uniform(0.115, 0, 2 * s),
 sd_slope = uniform(0.004, 0, 2 * s), P1 = diag(0.1, 2),
 distribution = "poisson")


result <- kfilter(model)
ts.plot(cbind(model$y, result$att[,1]), col = 1:2)
legend("topleft", legend = c("y", "est_level"), col = 1:2, lty = 1)
