library("KFAS")
library("bssm")
data("GlobalTemp", package = "KFAS")
model_temp <- ssm_mlg(GlobalTemp, H = matrix(c(0.15,0.05,0, 0.05), 2, 2),
  R = 0.05, Z = matrix(1, 2, 1), T = 1, P1 = 10,
  state_names = "temperature",
  # using default values, but being explicit for testing purposes
  D = matrix(0, 2, 1), C = matrix(0, 1, 1))
# result <- kfilter(model_temp)
z <-matrix(1, 2, 1)
h<-matrix(c(0.15,0.05,0, 0.05), 2, 2)

# benchmark("kfilter" = {result <- kfilter(model_temp)},
#
#           replications = 100,
#           columns = c("test", "replications", "elapsed",
#                        "user.self", "sys.self"))

total_time <- 0
for (x in 1:10) {
    start_time <- Sys.time()
    result <- kfilter(model_temp)
    end_time <- Sys.time()
    total_time <- total_time + end_time - start_time
}
print(total_time/10)