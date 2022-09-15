using NNE.ExactCalculation
import BSON: @save
using Base.Iterators

exp_spaced_values(min_value, max_value, n) = exp.((LinRange(log(min_value), log(max_value), n)))

function calc_losses(s_values, t_values, σ_values; problem=construct_problem())
    map(x -> get_mean_time_integrated_loss(x..., problem), product(s_values, t_values, σ_values))
end

function save_defaults()
    min_s = 0.01
    max_s = 10000
    num_s = 1000
    s_values = exp_spaced_values(min_s, max_s, num_s)
    t_values = [1, 2, 4, 8, 16]
    sigma = 1.0
    losses = reshape(calc_losses(s_values, t_values, [sigma]; problem), length(s_values), length(t_values))
    @save "results/exact_linear_perceptron.bson" s_values t_values sigma problem losses
end
