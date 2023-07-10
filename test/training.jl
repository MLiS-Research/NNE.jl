using TestItems


@testitem "Test Simulated Annealing MNIST training" begin
    using NNE
    using NNE.ImageClassification
    using NNE.Experiments
    import Flux
    using Random

    Random.seed!(1234) # Use a random seed
    for device in [Flux.cpu, Flux.gpu]
        model = generate_image_model(:MNIST; device, outputs=10)
        config = PreprocessConfig(shuffle=true, max_samples_per_label=16)
        dataset = load_dataset(:MNIST; split=SplitTest, device, config)

        alg_config = AlgorithmConfig(0.0, Float32(0.05), 0.25) # Always accept
        experiment_config = ExperimentConfig(1, 10, alg_config, false)

        results = NNE.Experiments.run(experiment_config, model, dataset)
        observations = results[:observations]
        activity = diff(observations) .!= zero(eltype(observations))
        @test all(activity) # Should change every time
    end
end
@testitem "Test Trajectory Ensemble MNIST training" begin
    using NNE
    using NNE.ImageClassification
    using NNE.Experiments
    import Flux
    using Random

    Random.seed!(1234) # Use a random seed
    for device in [Flux.cpu, Flux.gpu]
        model = generate_image_model(:MNIST; device, outputs=10)
        config = PreprocessConfig(shuffle=true, max_samples_per_label=16)
        dataset = load_dataset(:MNIST; split=SplitTest, device, config)

        alg_config = AlgorithmConfig(0.0, Float32(0.05), 0.25) # Always accept
        trajectory_length = 4
        experiment_config = ExperimentConfig(trajectory_length, 10, alg_config, false)

        results = NNE.Experiments.run(experiment_config, model, dataset)
        observations = results[:observations]
        activity = diff(observations) .!= zero(eltype(observations))
        @test all(activity) # Should change every time
    end
end