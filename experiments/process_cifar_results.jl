using NNE
using Experimenter
using Flux
using MLDatasets
using NNE.ImageClassification
using NNE.Experiments
using Random
using Dates
using ProgressBars
using BSON



db = open_db("experiments.db", joinpath(pwd(), "results", "overfitting"));
trials = [t for t in get_trials_by_name(db, "CIFAR") if t.has_finished && t.configuration[:tau] == 96];
dataset_name = :CIFAR10;
device = Flux.gpu;
Random.seed!(46238723); # Use a random seed
model = generate_image_model(dataset_name; device, outputs=10);
pre_process_config = PreprocessConfig(shuffle=true, max_samples_per_label=200);
dataset = load_dataset(dataset_name; split=SplitTrain, device, config=pre_process_config);
test_dataset = load_dataset(dataset_name; split=SplitTest, device);

ensembles = map(trials) do t
    ps = Flux.gpu.(t.results[:final_state])
    NNE.Ensembles.ClassificationEnsemble(model, ps)
end;

ensemble_predictions = map(ensembles) do ensemble
    NNE.Interfaces.predict(ensemble, test_dataset.features)
end;

ensemble_accuracies = map(ensemble_predictions) do preds
    sum(preds .== test_dataset.labels) / length(test_dataset.labels)
end;

ensemble_train_accuracies = map(ensembles) do ensemble
    preds = NNE.Interfaces.predict(ensemble, dataset.features)
    sum(preds .== dataset.labels) / length(dataset.labels)
end;

