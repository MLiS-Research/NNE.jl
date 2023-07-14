using NNE
using NNE.ImageClassification
using NNE.Experiments
import Flux
using Random

Random.seed!(89242464234); # Use a random seed
device = Flux.gpu;
model = generate_image_model(:MNIST; device, outputs=10);
config = PreprocessConfig(shuffle=true, max_samples_per_label=200);
dataset = load_dataset(:MNIST; split=SplitTrain, device, config); # Get a way to get a validation set
validation_dataset = load_dataset(:MNIST; split=SplitTest, device);

alg_config = AlgorithmConfig(150.0, Float32(0.05), 0.25); # Always accept
tb_config = TensorboardLoggingConfig(
    "results/logging/overfitting/nne_trial_1",
    100,
    2000,
    2000,
    2000
)
experiment_config = ExperimentConfig(32, 10_000, alg_config, true, tb_config);

results = NNE.Experiments.run(experiment_config, model, dataset; validation_dataset);
observations = results[:observations] ./ experiment_config.trajectory_length