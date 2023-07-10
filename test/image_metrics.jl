using TestItems


@testitem "Test MNIST loss calculation on CPU" begin
    using NNE
    using NNE.ImageClassification
    import NNE.Utils
    import Flux

    device = Flux.cpu
    model = generate_image_model(:MNIST; device, outputs=10)
    config = PreprocessConfig(shuffle=true, max_samples_per_label=16)
    dataset = load_dataset(:MNIST; split=SplitTest, device, config)
    loss_fn = NNE.Utils.FluxCrossEntropyLossObservable(model, dataset)

    @test typeof(loss_fn(model.parameters)) <: Any
end
@testitem "Test MNIST loss calculation on GPU" begin
    using NNE
    using NNE.ImageClassification
    import Flux

    device = Flux.gpu
    gpu_model = generate_image_model(:MNIST; device, outputs=10)
    config = PreprocessConfig(shuffle=true, max_samples_per_label=16)
    gpu_dataset = load_dataset(:MNIST; split=SplitTest, device, config)
    gpu_loss_fn = NNE.ImageClassification.ImageCrossEntropyLossObservable(gpu_model, gpu_dataset)

    gpu_loss_single = gpu_loss_fn(gpu_model.parameters)
    trajectory_length = 4
    gpu_loss_trajectory = gpu_loss_fn([gpu_model.parameters for _ in 1:trajectory_length])

    @test all(gpu_loss_trajectory .≈ [gpu_loss_single for _ in 1:trajectory_length])

    device = Flux.cpu
    model = generate_image_model(:MNIST; device, outputs=10)
    dataset = load_dataset(:MNIST; split=SplitTest, device, config)
    loss_fn = NNE.ImageClassification.ImageCrossEntropyLossObservable(model, dataset)

    loss_single = loss_fn(model.parameters)

    @test gpu_loss_single ≈ loss_single
end

@testitem "Test MNIST accuracy on CPU" begin
    using NNE
    using NNE.ImageClassification
    import Flux

    device = Flux.cpu
    model = generate_image_model(:MNIST; device, outputs=10)
    config = PreprocessConfig(shuffle=true, max_samples_per_label=16)
    dataset = load_dataset(:MNIST; split=SplitTest, device, config)
    accuracy_fn = NNE.ImageClassification.generate_ensemble_accuracy_fn(dataset, model)

    trajectory_length = 4
    @test accuracy_fn([model.parameters for _ in 1:trajectory_length]) ≈ accuracy_fn(model.parameters)
end
@testitem "Test MNIST accuracy calculation on GPU" begin
    using NNE
    using NNE.ImageClassification
    import Flux

    device = Flux.gpu
    gpu_model = generate_image_model(:MNIST; device, outputs=10)
    config = PreprocessConfig(shuffle=true, max_samples_per_label=16)
    gpu_dataset = load_dataset(:MNIST; split=SplitTest, device, config)
    gpu_accuracy_fn = NNE.ImageClassification.generate_ensemble_accuracy_fn(gpu_dataset, gpu_model)

    gpu_accuracy_single = gpu_accuracy_fn(gpu_model.parameters)
    trajectory_length = 4
    gpu_accuracy_trajectory = gpu_accuracy_fn([gpu_model.parameters for _ in 1:trajectory_length])

    @test gpu_accuracy_trajectory ≈ gpu_accuracy_single

    device = Flux.cpu
    model = generate_image_model(:MNIST; device, outputs=10)
    dataset = load_dataset(:MNIST; split=SplitTest, device, config)
    accuracy_fn = NNE.ImageClassification.generate_ensemble_accuracy_fn(dataset, model)

    @test accuracy_fn(model.parameters) ≈ gpu_accuracy_single
end

