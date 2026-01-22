using TestItems


@testitem "Test MNIST loss calculation on CPU" begin
    using NNE
    using NNE.ImageClassification
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
    gpu_loss_fn = NNE.Utils.FluxCrossEntropyLossObservable(gpu_model, gpu_dataset)

    gpu_loss_single = gpu_loss_fn(gpu_model.parameters)

    device = Flux.cpu
    model = generate_image_model(:MNIST; device, outputs=10)
    dataset = load_dataset(:MNIST; split=SplitTest, device, config)
    loss_fn = NNE.Utils.FluxCrossEntropyLossObservable(model, dataset)

    loss_single = loss_fn(model.parameters)

    @test gpu_loss_single ≈ loss_single
end
@testitem "Test MNIST GPU predictions" begin
    using NNE
    using NNE.ImageClassification
    import Flux
    import NNE.Interfaces: predict, features, labels

    device = Flux.gpu
    gpu_model = generate_image_model(:MNIST; device, outputs=10)
    config = PreprocessConfig(shuffle=true, max_samples_per_label=16)
    gpu_dataset = load_dataset(:MNIST; split=SplitTest, device, config)

    gpu_predictions = NNE.Interfaces.predict(gpu_model, features(gpu_dataset))

    device = Flux.cpu
    model = generate_image_model(:MNIST; device, outputs=10)
    dataset = load_dataset(:MNIST; split=SplitTest, device, config)

    cpu_predictions = NNE.Interfaces.predict(model, features(dataset))

    @test all(Array(gpu_predictions) .== cpu_predictions)
end
@testitem "Test MNIST accuracy on CPU" begin
    using NNE
    using NNE.ImageClassification
    import Flux
    import NNE.Interfaces: predict, features, labels

    device = Flux.cpu
    model = generate_image_model(:MNIST; device, outputs=10)
    config = PreprocessConfig(shuffle=true, max_samples_per_label=16)
    dataset = load_dataset(:MNIST; split=SplitTest, device, config)

    predictions = NNE.Interfaces.predict(model, features(dataset))

    @test typeof(NNE.Utils.accuracy(labels(dataset), predictions)) <: Any
end