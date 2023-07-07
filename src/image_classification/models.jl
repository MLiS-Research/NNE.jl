using Flux

struct ImageModel{T, RE}
    parameters::T
    reconstruct_fn::RE
    num_outputs::Int
end

function generate_image_model(model_name::Symbol; outputs::Int, device=Flux.cpu, seed::Int=948679378, kwargs...)
    previous_state = copy(Random.default_rng())
    Random.seed!(seed)
    model = create_model(Val(model_name); outputs, device, kwargs...)
    copy!(Random.default_rng(), previous_state)
    ps, re = Flux.destructure(model)
    return ImageModel(ps, re, outputs)
end

# Allows for overriding with custom implementations for different symbols
create_model(model_name::Val{:MNIST}; kwargs...) = generate_mnist_model(; kwargs...)
create_model(model_name::Val{:CIFAR10}; kwargs...) = generate_cifar10_model(; kwargs...)

function generate_mnist_model(; outputs=10, device=cpu)
    model = Flux.Chain(
        Flux.Conv((5, 5), 1 => 16, pad=(1, 1), Flux.relu), # Operates on 28x28
        x -> Flux.maxpool(x, (2, 2)),
        Flux.Conv((3, 3), 16 => 8, pad=(1, 1), Flux.relu; stride=2), #Operates on 13x13
        x -> Flux.maxpool(x, (2, 2)),
        x -> reshape(x, :, size(x, 4)), # Output should be (3,3,8,N)
        Flux.Dense(72, outputs)
    ) |> device
    # Total Parameters: 2306
    # Size: 9.742KiB
    return model
end
function generate_cifar10_model(; outputs=10, device=cpu)
    model = Flux.Chain(
        Flux.Conv((5, 5), 3 => 16, pad=(1, 1), Flux.relu), # Operates on 28x28
        x -> Flux.maxpool(x, (2, 2)),
        Flux.Conv((3, 3), 16 => 8, pad=(1, 1), Flux.relu; stride=2), #Operates on 13x13
        x -> Flux.maxpool(x, (2, 2)),
        x -> reshape(x, :, size(x, 4)),
        Flux.Dense(128, outputs)
    ) |> device
    # Total Parameters: 3666
    # Size: 15.055KiB
    return model
end

export generate_image_model