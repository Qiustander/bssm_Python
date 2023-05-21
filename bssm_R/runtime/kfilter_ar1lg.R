library('bssm')
n <- 200
mu <- 2
rho <- 0.7
sd_y <- 0.1
sigma <- 0.5
beta <- -1
x <- rnorm(n)
z <- y <- numeric(n)
z[1] <- rnorm(1, mu, sigma / sqrt(1 - rho^2))
y[1] <- rnorm(1, beta * x[1] + z[1], sd_y)
for(i in 2:n) {
  z[i] <- rnorm(1, mu * (1 - rho) + rho * z[i - 1], sigma)
  y[i] <- rnorm(1, beta * x[i] + z[i], sd_y)
}
model <- ar1_lg(y, rho = uniform(0.5, -1, 1),
  sigma = halfnormal(1, 10), mu = normal(0, 0, 1),
  sd_y = halfnormal(1, 10),
  xreg = x,  beta = normal(0, 0, 1))

total_time <- 0
for (x in 1:10) {
    start_time <- Sys.time()
    result <- kfilter(model)
    end_time <- Sys.time()
    total_time <- total_time + end_time - start_time
}
print(total_time/10)