using Random

function generate_data(Dx, Dy, N, noise; rng=Random.GLOBAL_RNG)
    X = rand(rng, Dx + 1, N)
    X[Dx+1, :] .= 1 # Transform into X̃ to include biases
    w = (rand(rng, Dy, Dx + 1) .* 2 .- 1)
    y = w * X + (rand(rng, Dy, N) .* noise) .* 2 .- 1 # Generate a random straight line with some noise
    return X, y, w
end

function generate_weights(rng, Dx, Dy)
    # W includes the biases as well
    rand(rng, Dy, Dx + 1)
end

@inline function predict(W, X, y)
    W * X + y
end

function generate_seeded_data(; seed=1234, Dx=1, Dy=1, N=256, noise=0.0)
    rng = Random.MersenneTwister(seed)
    x, y, w = generate_data(Dx, Dy, N, noise; rng=rng)
    return x, y, w
end

loss(W, x, y) = sum((W * x - y) .^ 2) / 2 ./ size(x, 2)