using Flux
using MLDatasets
using NNE
using NNE.ImageClassification
using NNE.Experiments
using Random
using Dates
using ProgressBars
using BSON


dataset_name = :CIFAR10
# optimiser = :adam
epochs = 500
# lr = 0.001
use_progress = true
logging_dir = "cifar_gd_$optimiser"
config = Dict{Symbol,Any}(
    :samples_per_label => 200,
    :dataset_name => dataset_name,
    :device => :gpu,
    :epochs => epochs,
    :seed => 46238723,
    :use_progress => use_progress,
    :logging_dirname => logging_dir
)
Random.seed!(config[:seed]) # Use a random seed
device = Flux.gpu
model = generate_image_model(dataset_name; device, outputs=10)
pre_process_config = PreprocessConfig(shuffle=true, max_samples_per_label=config[:samples_per_label])
dataset = load_dataset(dataset_name; split=SplitTrain, device, config=pre_process_config) # Get a way to get a validation set
test_dataset = load_dataset(dataset_name; split=SplitTest, device)

X = dataset.features
y = dataset.labels
m = model.model


function create_accuracy_fn(model)
    function acc(x, y)
        logits = Array(model(x))
        preds = [k[1] for k in reshape(argmax(logits, dims=1), :)]
        return sum(preds .== y) / length(y) * 100
    end
end


validation_size = Int(0.3 * length(y))
X_train = @views X[:, :, :, validation_size+1:end]
y_train = @views y[validation_size+1:end]
X_validation = @views X[:, :, :, 1:validation_size]
y_validation = @views y[1:validation_size]

y_onehot_train = Flux.onehotbatch(y_train, 1:10)
y_onehot = Flux.onehotbatch(y, 1:10)
batch_size = 64
loader_train = Flux.DataLoader((X_train, y_onehot_train); batchsize=batch_size, shuffle=true);
loader_full = Flux.DataLoader((X, y_onehot); batchsize=batch_size, shuffle=true);

function train_model!(model, loader, epochs, validation_gpu_features, validation_cpu_labels; lr=0.001, use_progress=true, validation_freq=10, optimiser=:adam)
    acc_fn = create_accuracy_fn(model)
    best_model = nothing
    best_epoch = 1
    best_accuracy = -Inf32
    losses = Float32[]
    accuracies = Float64[]
    iter = use_progress ? ProgressBar(1:epochs) : (1:epochs)
    _optim = if optimiser == :adam
        Flux.Adam(lr)
    elseif optimiser == :sgd
        Flux.Descent(lr)
    else
        error("Unknown optimiser: $optimiser")
    end
    optim = Flux.setup(_optim, model)  # will store optimiser momentum, etc.
    for epoch in iter
        total_loss = 0.0
        num_samples = 0
        for (x, y) in loader
            loss, grads = Flux.withgradient(model) do _m
                # Evaluate model and loss inside gradient context:
                y_hat = _m(x)
                Flux.logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, model, grads[1])
            num_samples += 1
            total_loss += loss
        end
        push!(losses, total_loss / num_samples)  # logging, outside gradient context

        if epoch % validation_freq == 0
            accuracy = acc_fn(validation_gpu_features, validation_cpu_labels)
            if use_progress
                set_multiline_postfix(iter, "Accuracy: $(round(accuracy, sigdigits=4))%\nLoss: $(round(last(losses), sigdigits=4))")
            end
            if accuracy > best_accuracy
                best_epoch = epoch
                best_accuracy = accuracy
                best_model = deepcopy(model)
            end
            push!(accuracies, accuracy)
        end
    end
    info = Dict{Symbol,Any}(
        :best_model => best_model,
        :losses => losses,
        :epochs => epochs,
        :best_epoch => best_epoch,
        :optimiser => optimiser,
        :lr => lr,
        :validation_freq => validation_freq,
        :accuracies => accuracies
    )

    return best_model, info
end

num_models = 96;
seeds = [(Int(rand(UInt32)) % 232304 + 1233) for _ in 1:num_models];
models = [generate_image_model(dataset_name; device, outputs=10, seed=s) for s in seeds];
validation_freq = 1
pre_training_results = map(ProgressBar(models)) do m
    return train_model!(m.model, loader_train, epochs, X_validation, y_validation |> Flux.cpu; lr, use_progress=false, optimiser, validation_freq)
end;
# Re-generate the models
models = [generate_image_model(dataset_name; device, outputs=10, seed=s) for s in seeds];
training_results = map(ProgressBar(collect(zip(models, pre_training_results)))) do arg
    m, (_, training_result) = arg
    _epochs = training_result[:best_epoch]
    return train_model!(m.model, loader_full, _epochs, nothing, nothing; lr, use_progress=false, optimiser, validation_freq=Inf)
end
best_models = [m.model for m in models];
training_infos = [r[2] for r in pre_training_results];
best_accuracies = [maximum(info[:accuracies]) for info in training_infos]

function create_ensemble(base_model, flux_models)
    parameters = map(flux_models) do m
        ps, _ = Flux.destructure(m)
        return ps
    end
    ensemble = NNE.Ensembles.ClassificationEnsemble(base_model, parameters)
    return ensemble
end

is_loading = false
if is_loading
    BSON.@load "results/gd_data_cifar10_$optimiser.bson" parameters infos
    gd_ensemble = NNE.Ensembles.ClassificationEnsemble(model, Flux.gpu.(parameters))
else
    gd_ensemble = create_ensemble(model, best_models)
end

is_saving = true
if is_saving
    parameters = Flux.cpu.([m.parameters for m in gd_ensemble.models])
    infos = [Dict{Symbol,Any}(:accuracies => i[:accuracies], :losses => i[:losses] .* batch_size) for i in training_infos]
    BSON.@save "results/gd_data_cifar10_$optimiser.bson" parameters infos
end

gd_ensemble_predictions = NNE.Interfaces.predict(gd_ensemble, test_dataset.features);
gd_ensemble_accuracy = sum(gd_ensemble_predictions .== test_dataset.labels) / length(test_dataset.labels)
@show gd_ensemble_accuracy

using Plots
begin
    plt = plot()
    for info in training_infos
        plot!(plt, info[:losses], label=nothing, alpha=0.2)
    end
    xlabel!(plt, "Epochs")
    ylabel!(plt, "Loss")
    Plots.savefig(plt, "./tmp/loss_$optimiser.pdf")
    nothing
end

begin
    plt = plot()
    for info in training_infos
        plot!(plt, info[:accuracies], label=nothing, alpha=0.2)
    end
    xlabel!(plt, "Epochs")
    ylabel!(plt, "Accuracy")
    Plots.savefig(plt, "./tmp/accuracy_$optimiser.pdf")
    nothing
end