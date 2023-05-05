library("KFAS")
library("bssm")
data("GlobalTemp", package = "KFAS")
model_temp <- ssm_mlg(GlobalTemp, H = matrix(c(0.15,0.05,0, 0.05), 2, 2),
  R = 0.05, Z = matrix(1, 2, 1), T = 1, P1 = 10,
  state_names = "temperature",
  # using default values, but being explicit for testing purposes
  D = matrix(0, 2, 1), C = matrix(0, 1, 1))
result <- kfilter(model_temp)
z <-matrix(1, 2, 1)
h<-matrix(c(0.15,0.05,0, 0.05), 2, 2)

esty <- result$att%*%t(z)+ matrix( rnorm(length(model_temp$y[,1])*2,mean=0,sd=1), length(model_temp$y[,1]), 2)%*%h 
# ts.plot(cbind(result$att), col = 1:1)
# legend("bottomright", legend = c("est_temp"), col = 1:1, lty = 1)

ts.plot(cbind(esty[,2], model_temp$y[,2]), col = 1:2)
legend("bottomright", legend = c("est_y", "y"), col = 1:2, lty = 1)


