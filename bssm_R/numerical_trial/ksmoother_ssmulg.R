library('bssm')
n <- 100
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

# three states, level, b1, b2!
model <- ssm_ulg(y, Z, H, T, R, a1, P1,
                 init_theta = c(1, 0.1, 0.1),
                 update_fn = update_fn, prior_fn = prior_fn,
                 state_names = c("level", "b1", "b2"),
                 # using default values, but being explicit for testing purposes
                 C = matrix(0, 3, 1), D = numeric(1))

result_fast <- fast_smoother(model)
esty <- result_fast[,1] + result_fast[,2] * x1 + b2 * result_fast[,3] # ts.plot(cbind(b1, b2, result$att[,2], result$att[,3]), col = 1:5)

# ts.plot(cbind(rep(1, 100), b1, b2, result_fast[,1], result_fast[,2], result_fast[,3]), col = 1:6)
# legend("bottomright", legend = c("level", "b1", "b2", "est_level", "est_b1", "est_b2"), col = 1:6, lty = 1)
ts.plot(cbind(esty, y), col = 1:2)
legend("bottomright", legend = c("esty", "y"), col = 1:2, lty = 1)

# normal 
result <- smoother(model)
esty <- result$alphahat[,1] + result$alphahat[,2] * x1 + b2 * result$alphahat[,3] 

ts.plot(cbind(esty, y), col = 1:2)
legend("bottomright", legend = c("esty", "y"), col = 1:2, lty = 1)

# the same performance