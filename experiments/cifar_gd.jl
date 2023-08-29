using Flux
using MLDatasets
using NNE
using NNE.ImageClassification
using NNE.Experiments
using Random
using Dates
using ProgressBars


dataset_name = :CIFAR10
epochs = 1000
use_progress = true
logging_dir = "cifar_gd"
config = Dict{Symbol,Any}(
    :samples_per_label => 200,
    :dataset_name => dataset_name,
    :device => :gpu,
    :epochs => epochs,
    :seed => 46938723,
    :use_progress => use_progress,
    :logging_dirname => logging_dir
)
Random.seed!(config[:seed]) # Use a random seed
device = Flux.gpu
model = generate_image_model(dataset_name; device, outputs=10)
pre_process_config = PreprocessConfig(shuffle=true, max_samples_per_label=config[:samples_per_label])
dataset = load_dataset(dataset_name; split=SplitTrain, device, config=pre_process_config) # Get a way to get a validation set
validation_dataset = load_dataset(dataset_name; split=SplitTest, device)

X = dataset.features
y = dataset.labels

y_onehot = Flux.onehotbatch(y, 1:10)

m = model.model

function create_accuracy_fn(model)
    function acc(x, y)
        logits = Array(model(x))
        preds = [k[1] for k in reshape(argmax(logits, dims=1), :)]
        return sum(preds .== y) / length(y) * 100
    end
end



loader = Flux.DataLoader((X, y_onehot), batchsize=64, shuffle=true);

function train_model!(model, loader, epochs, validation_gpu_features, validation_cpu_labels; lr=0.001, use_progress=true, validation_freq=25)
    acc_fn = create_accuracy_fn(model)
    best_model = nothing
    best_accuracy = -Inf32
    losses = Float32[]
    accuracies = Float64[]
    iter = use_progress ? ProgressBar(1:epochs) : (1:epochs)
    optim = Flux.setup(Flux.Adam(lr), model)  # will store optimiser momentum, etc.
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
            num_samples += length(y)
            total_loss += loss
        end
        push!(losses, total_loss / num_samples)  # logging, outside gradient context

        if epoch % validation_freq == 0
            accuracy = acc_fn(validation_gpu_features, validation_cpu_labels)
            if use_progress
                set_multiline_postfix(iter, "Accuracy: $(round(accuracy, sigdigits=4))%\nLoss: $(round(last(losses), sigdigits=4))")
            end
            if accuracy > best_accuracy
                best_accuracy = accuracy
                best_model = deepcopy(model)
            end
            push!(accuracies, accuracy)
        end
    end
    info = Dict{Symbol,Any}(
        :best_model => best_model,
        :losses => losses,
        :accuracies => accuracies
    )
    return best_model, info
end


validation_gpu_features = validation_dataset.features;
validation_cpu_labels = Array(validation_dataset.labels);
num_models = 96;
seeds = [(Int(rand(UInt32)) % 232304 + 1233) for _ in 1:num_models];
models = [generate_image_model(dataset_name; device, outputs=10, seed=s) for s in seeds];

training_results = map(ProgressBar(models)) do m
    return train_model!(m.model, loader, 200, validation_gpu_features, validation_cpu_labels; lr=0.001, use_progress=false)
end;
best_models = [r[1] for r in training_results];
training_infos = [r[2] for r in training_results];
best_accuracies = [maximum(info[:accuracies]) for info in training_infos]

function create_ensemble(base_model, flux_models)
    parameters = map(flux_models) do m
        ps, _ = Flux.destructure(m)
        return ps
    end
    ensemble = NNE.Ensembles.ClassificationEnsemble(base_model, parameters)
    return ensemble
end

gd_ensemble = create_ensemble(model, best_models);

gd_ensemble_predictions = NNE.Interfaces.predict(gd_ensemble, validation_dataset.features);
gd_ensemble_accuracy = sum(gd_ensemble_predictions .== validation_dataset.labels) / length(validation_dataset.labels)