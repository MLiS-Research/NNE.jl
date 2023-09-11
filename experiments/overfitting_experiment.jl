using NNE
using NNE.ImageClassification
using NNE.Experiments
import Flux
using Random
using Dates


function run_experiment(config, trial_id)
    seed = config[:seed]
    samples_per_label = config[:samples_per_label]
    dataset_name = config[:dataset_name]
    sigma = config[:sigma]
    s = config[:s]
    tau = config[:tau]
    epochs = config[:epochs]
    param_frac_changed = config[:param_frac_changed]
    use_progress = haskey(config, :use_progress) ? config[:use_progress] : false
    log_dirname = haskey(config, :logging_dirname) ? config[:logging_dirname] : "overfitting"
    time_identifier = replace(string(now()), ":" => "-")[begin:end-4] * "-id-$(string(rand(100000:999999)))"

    Random.seed!(seed) # Use a random seed
    device = Flux.gpu
    model = generate_image_model(dataset_name; device, outputs=10)
    pre_process_config = PreprocessConfig(shuffle=true, max_samples_per_label=samples_per_label)
    dataset = load_dataset(dataset_name; split=SplitTrain, device, config=pre_process_config) # Get a way to get a validation set
    test_dataset = load_dataset(dataset_name; split=SplitTest, device)

    alg_config = AlgorithmConfig(s, Float32(sigma), param_frac_changed) # Always accept

    tb_config = TensorboardLoggingConfig(
        "results/logging/$(log_dirname)/$(string(dataset_name))_$(time_identifier)",
        1000,
        4000,
        4000,
        4000
    )
    experiment_config = ExperimentConfig(tau, epochs, alg_config, use_progress, tb_config)

    results = NNE.Experiments.run(experiment_config, model, dataset; test_dataset)

    return results
end