module ToyClassificationProblem
include("classification_data.jl")
using ..ProblemGenerators
using TPS.MetropolisHastings
using Random

function generate_toy_dataset(n_samples::Int, seed::Int=67859863)
    rng = Random.MersenneTwister(seed) # Have a preset RNG for this function
    data = rand(rng, 2, n_samples) .* 2 .- 1 # Dimension 1 is the data dim, 2 is the number of samples
    labels = (Int8.(vec((data[1, :] .* 0.8 - data[1, :] .* 0.2 .- 0.2) .< data[2, :]).+1)) # Classes are either 1 or 2
    labels_one_hot = zeros(2, n_samples)
    for i in eachindex(labels)
        labels_one_hot[labels[i], i] = 1
    end
    
    return TPSClassificationDataset(data, labels, labels_one_hot)
end

function create_toy_model(dims=2, num_classes=2)
    return Flux.Chain(
        Flux.Dense(dims, dims*2, tanh),
        Flux.Dense(dims*2, dims*4, tanh),
        Flux.Dense(dims*4, dims*4, tanh),
        Flux.Dense(dims*4, dims, tanh),
        Flux.Dense(dims, num_classes, identity)
    )
end

function construct_toy_problem(τ, σ=1.0; n_samples::Int=128, data_seed::Int=67859863, rng=Random.GLOBAL_RNG)
    mdl = create_toy_model(2,2)
    dataset = generate_toy_dataset(n_samples, data_seed)
    state, _ = Flux.destructure(mdl)
    loss_fn = construct_cross_entropy_loss_fn(mdl, dataset)
    
    return create_problem(loss_fn, state, τ, σ; rng=rng)
end

function construct_algorithm(τ, s, σ)
    if τ==1
        return MetropolisHastings.gaussian_sa_algorithm(s, σ)
    else
        return MetropolisHastings.gaussian_trajectory_algorithm(s, σ; max_width = 1)
    end
end

export construct_toy_problem, construct_algorithm

end