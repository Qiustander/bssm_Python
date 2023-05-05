library("bssm")
# x <- cumsum(rnorm(20))
# y <- x + rnorm(20, sd = 0.1)
# model <- bsm_lg(y, sd_level = 1, sd_y = 0.1)
# g<-kfilter(model)
# ts.plot(cbind(y, x, kfilter(model)$att), col = 1:3)

data("nhtemp", package = "datasets")
prior <- halfnormal(1, 10)
model <- bsm_lg(y=nhtemp, sd_y = prior, sd_level = prior, sd_slope = prior)
result <- kfilter(model)
ts.plot(cbind(model$y, result$att[,1]), col = 1:3)
legend("bottomright", legend = c("y", "level"), col = 1:3, lty = 1)


############## Case 2
# set.seed(1)
# n <- 50
# x <- rnorm(n)
# level <- numeric(n)
# level[1] <- rnorm(1)
# for (i in 2:n) level[i] <- rnorm(1, -0.2 + level[i-1], sd = 0.1)
# y <- rnorm(n, 2.1 + x + level)
# model <- bsm_lg(y, sd_y = halfnormal(1, 5), sd_level = 0.1, a1 = level[1],
#   P1 = matrix(0, 1, 1), xreg = x, beta = normal(1, 0, 1),
#   D = 2.1, C = matrix(-0.2, 1, 1))

# result <- kfilter(model)
# ts.plot(cbind(level, result$att), col = 1:2)
# legend("bottomright", legend = c("level", "est_level"), col = 1:2, lty = 1)

# esty <- rnorm(n, 2.1 + x + result$att)
# ts.plot(cbind(esty, y), col = 1:2)
# legend("bottomright", legend = c("esty", "y"), col = 1:2, lty = 1)