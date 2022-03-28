module MNISTTraining

using Flux
using MLDatasets
using TPS
using TPS.SimulatedAnnealing
using TPS.MetropolisHastings
using TPS.DiscreteTrajectory
using TPS.Convergence
using Random
using ProgressBars
using Plots

export generate_mnist_dataset, generate_mnist_model, get_mnist_testing_dataset, solve_mnist_sa, solve_mnist_sa_automatic, solve_mnist_trajectory, solve_mnist_trajectory_automatic

function generate_mnist_model(;outputs=10, device=cpu)
    model = Flux.Chain(
        Flux.Conv((5, 5), 1=>16, pad=(1,1), Flux.relu), # Operates on 28x28
        x -> Flux.maxpool(x, (2,2)),
        Flux.Conv((3, 3), 16=>8, pad=(1,1), Flux.relu; stride=2), #Operates on 13x13
        x -> Flux.maxpool(x, (2,2)),
        x -> reshape(x, :, size(x, 4)), # Output should be (3,3,8,N)
        Flux.Dense(72, outputs)
        ) |> device
    
    return model
end

function generate_mnist_dataset(n_samples::Int; seed::Int=659863, device=cpu, outputs=10, kwargs...)
    rng = Random.MersenneTwister(seed) # Have a preset RNG for this function
    labels = MNIST.trainlabels()
    selector = labels .< outputs
    selected_indices = (collect(1:length(labels))[selector])
    samples = shuffle(rng, selected_indices)[1:n_samples]
    labels = (labels)[samples]
    data = reshape(MNIST.traintensor(Float32)[:, :, samples], 28, 28, 1, n_samples)
    labels_one_hot = Flux.onehotbatch(labels, 0:(outputs-1))
    return data |> device, labels |> device, labels_one_hot |> device
end

function get_mnist_testing_dataset(;device=cpu, outputs = 10)
    labels = MNIST.testlabels()
    selector = labels .< outputs
    labels = labels[selector] |> device
    data = reshape(MNIST.testtensor(Float32)[:,:, selector], 28, 28, 1, :) |> device
    return data, labels
end

function create_mnist_sa_problem(n_samples::Int = 1000; device=cpu, outputs=10, kwargs...)
    data, labels, labels_one_hot = generate_mnist_dataset(n_samples; device, outputs, kwargs...)
    model = generate_mnist_model(; device, outputs, kwargs...)

    θ, re = Flux.destructure(model)

    function loss_fn(state)
        mdl = re(state)
        return Flux.Losses.logitcrossentropy(mdl(data), labels_one_hot)/n_samples
    end

    info = Dict(:features=>data, :labels=>labels, :labels_one_hot=>labels_one_hot, :initial_state=>θ, :model_re_fn=>re, :loss_fn=>loss_fn)

    obs = TPS.SimpleObservable(loss_fn)
    return SAProblem(obs, θ), info
end

function create_mnist_trajectory_state_and_loss(τ, σ, n_samples; rng=Random.GLOBAL_RNG, kwargs...)
    _, info = create_mnist_sa_problem(n_samples;kwargs...)
    θ = info[:initial_state]
    loss_fn = info[:loss_fn]
    states = []
    push!(states, θ)
    previous_state = θ
    for t=2:τ
        next_state = similar(θ)
        randn!(rng, next_state)
        next_state .= next_state.*σ .+ previous_state
        push!(states, next_state)
        previous_state = next_state
    end

    function traj_loss_fn(state)
        return sum(loss_fn(s) for s in state)
    end

    obs = TPS.SimpleObservable(traj_loss_fn)
    problem = DTProblem(obs, states)

    info[:traj_loss_fn] = traj_loss_fn
    info[:initial_trajectory] = deepcopy(states)

    return problem, info
end


function solve_mnist_sa(;s=500.0, σ=0.001, epochs=100000, n_samples=4096, fraction_to_include=1.0, device=cpu, outputs=2, show_progress=false, kwargs...)
    problem, info = create_mnist_sa_problem(n_samples; device, outputs)
    alg = get_guassian_mh_alg(s, σ; fraction_to_include);
    iter = show_progress ? ProgressBar(1:epochs) : epochs
    solution = solve(problem, alg, iter)
    info[:solution] = solution
    info[:s] = s
    info[:σ] = σ
    info[:τ] = 1
    info[:epochs] = epochs
    info[:n_samples] = n_samples
    info[:outputs] = outputs
    info[:fraction_to_include] = fraction_to_include
    return info
end

function solve_mnist_sa_automatic(;s=200.0, σ=0.001, n_samples=2048, device=cpu, outputs=2, fraction_to_include=1.0, warmup_steps=0, polling_frequency=1, max_buffer_size=10000, relative_gradient_size=1e-4, relative_error_size=2e-2, max_epochs=100000)
    problem, info = create_mnist_sa_problem(n_samples; device, outputs)
    alg = get_guassian_mh_alg(s, σ; fraction_to_include=fraction_to_include);
    iteration_options = AutomaticConvergenceOptions(warmup_steps, polling_frequency, max_buffer_size, max_epochs, relative_gradient_size, relative_error_size)
    solution = solve(problem, alg, iteration_options)
    info[:solution] = solution
    info[:s] = s
    info[:σ] = σ
    info[:τ] = 1
    info[:fraction_to_include] = fraction_to_include
    info[:warmup_steps] = warmup_steps
    info[:polling_frequency] = polling_frequency
    info[:max_buffer_size] = max_buffer_size
    info[:max_epochs] = max_epochs
    info[:relative_gradient_size] = relative_gradient_size
    info[:relative_error_size] = relative_error_size
    return info
end


function solve_mnist_trajectory(;τ=4, s=50.0, σ=0.001, epochs=10000, n_samples=2048, device=cpu, outputs=2, show_progress=false, fraction_to_include=1.0, kwargs...)
    problem, info = create_mnist_trajectory_state_and_loss(τ, σ, n_samples; device, outputs)
    alg = MetropolisHastings.get_shooting_mh_alg(s, σ; fraction_to_include=fraction_to_include);
    iter = show_progress ? ProgressBar(1:epochs) : epochs
    solution = solve(problem, alg, iter)
    info[:solution] = solution
    info[:s] = s
    info[:σ] = σ
    info[:τ] = τ
    info[:epochs] = epochs
    info[:n_samples] = n_samples
    info[:outputs] = outputs
    info[:fraction_to_include] = fraction_to_include
    return info
end

function solve_mnist_trajectory_automatic(;τ=4,s=500.0, σ=0.05, n_samples=2048, device=cpu, outputs=2, fraction_to_include=1.0, warmup_steps=0, polling_frequency=1, max_buffer_size=10000, relative_gradient_size=1e-4, relative_error_size=2e-2, max_epochs=100000)
    problem, info = create_mnist_trajectory_state_and_loss(τ, σ, n_samples; device, outputs)
    alg = MetropolisHastings.get_shooting_mh_alg(s, σ; fraction_to_include=fraction_to_include);
    iteration_options = AutomaticConvergenceOptions(warmup_steps, polling_frequency, max_buffer_size, max_epochs, relative_gradient_size, relative_error_size)
    sol = solve(problem, alg, iteration_options)
    info[:solution] = sol
    info[:s] = s
    info[:σ] = σ
    info[:τ] = τ
    info[:fraction_to_include] = fraction_to_include
    info[:warmup_steps] = warmup_steps
    info[:polling_frequency] = polling_frequency
    info[:max_buffer_size] = max_buffer_size
    info[:max_epochs] = max_epochs
    info[:relative_gradient_size] = relative_gradient_size
    info[:relative_error_size] = relative_error_size
    return info
end

"""
Returns the accuracy of a model, based on input features and true labels. Returns as a % between 0 and 100.
"""
function test_accuracy(model, features, labels)
    pred = model(features)
    pred_labels = reshape([x[1]-1 for x in argmax(pred, dims=1)], :);
    return sum(pred_labels.==labels)/length(labels)*100
end

function get_activity(losses)
    return sum(diff(losses).!==zero(eltype(losses)))/(length(losses)-1)
end
end