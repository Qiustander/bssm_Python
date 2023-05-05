set.seed(1)
mu <- 2
rho <- 0.7
sd_y <- 0.1
sigma <- 0.5
beta <- -1
x <- rnorm(30)
z <- y <- numeric(30)
z[1] <- rnorm(1, mu, sigma / sqrt(1 - rho^2))
y[1] <- rnorm(1, beta * x[1] + z[1], sd_y)
for(i in 2:30) {
  z[i] <- rnorm(1, mu * (1 - rho) + rho * z[i - 1], sigma)
  y[i] <- rnorm(1, beta * x[i] + z[i], sd_y)
}
model <- ar1_lg(y, rho = uniform(0.5, -1, 1),
  sigma = halfnormal(1, 10), mu = normal(0, 0, 1),
  sd_y = halfnormal(1, 10),
  xreg = x,  beta = normal(0, 0, 1))

esty <- rep(0, length(x))
result <- kfilter(model)
for (i in 1:30){
esty[i] <- rnorm(1, beta * x[i] + result$att[i], sd_y)}
ts.plot(cbind(esty, y), col = 1:2)
legend("bottomright", legend = c("est_y", "y"), col = 1:2, lty = 1)

ts.plot(cbind(result$att, z), col = 1:2)
legend("bottomright", legend = c("est_z", "z"), col = 1:2, lty = 1)
