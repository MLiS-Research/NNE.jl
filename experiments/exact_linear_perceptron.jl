using NNE.ExactCalculation
import BSON: @save
using Base.Iterators

min_s = 0.01
max_s = 10000
num_s = 1000
s_values = exp.((LinRange(log(min_s), log(max_s), num_s)))
t_values = [1, 2, 4, 8, 16]
sigma = 1.0

problem = construct_problem()

losses = map(x->get_mean_time_integrated_loss(x..., sigma, problem), product(s_values, t_values))

@save "results/exact_linear_perceptron.bson" s_values t_values sigma problem losses
