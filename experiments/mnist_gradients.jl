include("run_helper.jl")
using CUDA
using Flux
using NNE.MNISTTraining
using NNE.Runner
using ProgressBars
using BSON: @save, @load

function train_mnist_model(;epochs=10000, lr=0.1, device=cpu, outputs=10, n_samples=4096, use_progress=true)
    info = Dict{Symbol, Any}(
        :epochs=>epochs,
        :lr=>lr,
        :device=>Symbol(device),
        :outputs=>outputs,
        :n_samples=>n_samples
    )

    data, labels, labels_one_hot = generate_mnist_dataset(n_samples; outputs, device);
    test_data, test_labels = get_mnist_testing_dataset(;device, outputs)
    labels = labels |> cpu # Labels should be on the CPU
    test_labels = test_labels |> cpu # Labels should be on the CPU
    model = generate_mnist_model(;outputs, device)

    loss() = Flux.Losses.logitcrossentropy(model(data), labels_one_hot)
    acccuracy() = Flux.mean(reshape((x->x[1]-1).(argmax(model(data) |> cpu, dims=1)), :) .== labels)
    test_acccuracy() = Flux.mean(reshape((x->x[1]-1).(argmax(model(test_data) |> cpu, dims=1)), :) .== test_labels)
    ps = params(model)
    opt = Descent(lr)
    losses = [loss()]
    accuracies = [acccuracy()]
    test_accuracies = [test_acccuracy()]
    iter = use_progress ? ProgressBar(1:epochs) : (1:epochs)
    for _ in iter
        gs = gradient(ps) do
            loss()
        end
        Flux.update!(opt, ps, gs)
        push!(losses, loss())
        push!(accuracies, acccuracy())
        push!(test_accuracies, test_acccuracy())
    end

    info[:losses] = [losses...]
    info[:accuracies] = [accuracies...]
    info[:test_accuracies] = [test_accuracies...]

    return info
end

function run_mnist_gradient_experiment(;kwargs...)
    info = train_mnist_model(;kwargs...)
    @save "results/mnist_gradient_experiment.bson" info
end