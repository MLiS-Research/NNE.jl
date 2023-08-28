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
acc_fn = create_accuracy_fn(m)


loader = Flux.DataLoader((X, y_onehot), batchsize=64, shuffle=true);
# 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64 OneHotMatrix)

optim = Flux.setup(Flux.Adam(0.0001), m);  # will store optimiser momentum, etc.

# Training loop, using the whole data set 1000 times:

validation_gpu_features = validation_dataset.features;
validation_cpu_labels = Array(validation_dataset.labels);
begin
    losses = Float32[]
    accuracies = Float64[]
    iter = ProgressBar(1:epochs)
    for epoch in iter
        for (x, y) in loader
            loss, grads = Flux.withgradient(m) do _m
                # Evaluate model and loss inside gradient context:
                y_hat = _m(x)
                Flux.logitcrossentropy(y_hat, y)
            end
            Flux.update!(optim, m, grads[1])
            push!(losses, loss)  # logging, outside gradient context
        end
        
        if epoch % 10 == 0
            accuracy = acc_fn(validation_gpu_features, validation_cpu_labels)
            set_multiline_postfix(iter, "Accuracy: $(round(accuracy, sigdigits=4))%\nLoss: $(round(last(losses), sigdigits=4))")
            push!(accuracies, accuracy)
        end
    end
end
