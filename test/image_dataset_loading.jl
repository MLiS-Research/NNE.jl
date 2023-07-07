using TestItems


@testitem "Test MNIST loading" begin
    using NNE
    using NNE.ImageClassification
    @test typeof(load_dataset(:MNIST; split=SplitTrain)) <: Any
    @test typeof(load_dataset(:MNIST; split=SplitTest)) <: Any
end

@testitem "Test CIFAR-10 loading" begin
    using NNE
    using NNE.ImageClassification
    @test typeof(load_dataset(:CIFAR10; split=SplitTrain)) <: Any
    @test typeof(load_dataset(:CIFAR10; split=SplitTest)) <: Any
end

@testitem "Test image max_labels - no shuffle" begin
    using NNE
    using NNE.ImageClassification
    num_samples_per_digit = 10
    config = PreprocessConfig(shuffle=false, max_samples_per_label=num_samples_per_digit)
    dataset = load_dataset(:MNIST; split=SplitTest, config)
    @test length(dataset.labels) == 10 * num_samples_per_digit # 10 images for each labels
    @test length(unique(dataset.labels)) == num_samples_per_digit
    @test extrema(dataset.labels) == (1, 10)
    @test all((count(x -> x == d, dataset.labels) == num_samples_per_digit for d in 1:10)) # Check each digit
end

@testitem "Test image max_labels - shuffle" begin
    using NNE
    using NNE.ImageClassification
    num_samples_per_digit = 10
    config = PreprocessConfig(shuffle=true, max_samples_per_label=num_samples_per_digit)
    dataset = load_dataset(:MNIST; split=SplitTest, config)
    @test length(dataset.labels) == 10 * num_samples_per_digit # 10 images for each labels
    @test length(unique(dataset.labels)) == num_samples_per_digit
    @test extrema(dataset.labels) == (1, 10)
    @test all((count(x -> x == d, dataset.labels) == num_samples_per_digit for d in 1:10)) # Check each digit
end

