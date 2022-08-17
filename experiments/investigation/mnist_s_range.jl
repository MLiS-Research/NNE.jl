using NNE
using NNE.MNISTTraining
using TPS
using TPS.MetropolisHastings
using Flux
using Memoization
using Statistics
using Plots

function create_problem_and_info(n_samples, τ, σ, device=:cpu)
    device_fn = device == :gpu ? gpu : cpu
    if τ==1
        return NNE.MNISTTraining.create_mnist_sa_problem(n_samples; device=device_fn, outputs=10)
    else
        return NNE.MNISTTraining.create_mnist_trajectory_state_and_loss(τ, σ, n_samples; device=device_fn, outputs=10)
    end
end

function get_algorithm(τ, s, σ, params_changed_frac=1.0, max_width=1)
    if τ == 1
        return gaussian_sa_algorithm(s, σ; params_changed_frac)
    elseif τ == 2
        return gaussian_trajectory_algorithm(s, σ; params_changed_frac, max_width)
    else
        return gaussian_trajectory_algorithm(s, σ; params_changed_frac, max_width, chance_to_shoot=(2 / τ))
    end
end

function plot_loss_and_acceptance(τ, s, σ, epochs, n_samples=1024, device=:cpu, params_changed_frac=1.0, max_width=1)
    problem, info = create_problem_and_info(n_samples, τ, σ, device)
    alg = get_algorithm(τ, s, σ, params_changed_frac, max_width)

    sol = solve(problem, alg, epochs)
    info[:final_state] = TPS.get_current_state(sol)

    accuracies = measure_train_accuracy(τ, info; device = (device == :gpu ? gpu : cpu), outputs=10);

    plt = plot(sol.observations ./ τ; xscale=:log10, yscale=:log10, legend=false)
    xlabel!("Epochs")
    ylabel!("L/τ")
    acceptance = round(1.0 - mean(Float64.(diff(sol.observations).==0)); sigdigits=4)
    title!("τ=$τ, s=$s,σ=$σ, n=2^$(Int(round(log2(n_samples)))), f=$params_changed_frac, <A>=$acceptance")
    return accuracies, plt
end

function reconstruct_mnist_models(τ, info_dict; outputs=10, device=cpu)
    model = generate_mnist_model(; outputs, device)
    _, re = Flux.destructure(model)
    if τ == 1
        model = re(info_dict[:final_state]) # reconstruct model from params
        return [model |> device]
    end

    models = [re(state) |> device for state in info_dict[:final_state]]
    return models
end

function measure_train_accuracy(τ, info_dict; device=cpu, outputs=10)
    models = reconstruct_mnist_models(τ, info_dict; outputs, device)
    features = info_dict[:features] |> device
    labels = info_dict[:labels] |> cpu
    accuracies = [Flux.mean(reshape((x -> x[1] - 1).(argmax(m(features) |> cpu, dims=1)), :) .== labels) for m in models]
    return accuracies
end
