using NNE
using NNE.ImageClassification
using NNE.Experiments
import Flux
using Random

Random.seed!(89242464234); # Use a random seed
device = Flux.gpu;
model = generate_image_model(:MNIST; device, outputs=10);
config = PreprocessConfig(shuffle=true, max_samples_per_label=200);
dataset = load_dataset(:MNIST; split=SplitTrain, device, config);

alg_config = AlgorithmConfig(150.0, Float32(0.05), 0.25); # Always accept
experiment_config = ExperimentConfig(64, 100_000, alg_config, true);

results = NNE.Experiments.run(experiment_config, model, dataset);
observations = results[:observations] ./ experiment_config.trajectory_length

using Plots
begin
    plt = plot(1:1000:length(observations), observations[1:1000:end], yscale=:log10, legend=false)
    xlabel!(plt, "Epochs")
    ylabel!(plt, "<L>/τ")
    return plt
end

import CUDA
ensemble = NNE.Ensembles.ClassificationEnsemble(model, cu.(results[:final_state]));
test_dataset = load_dataset(:MNIST; split=SplitTest, device);
test_predictions = NNE.Interfaces.predict(ensemble, NNE.Interfaces.features(test_dataset));
test_labels = NNE.Interfaces.labels(test_dataset);

accuracy = NNE.Utils.accuracy(test_labels, test_predictions)