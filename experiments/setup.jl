using Random

function generate_random_walk(; epochs, sigma, seed)
    rng = MersenneTwister(seed)

    positions = zeros(epochs)
    for i in 2:length(positions)-1
        positions[i] = positions[i-1] + randn(rng) * sigma
    end

    results = Dict{Symbol,Any}(
        :positions => positions,
        :epochs => epochs,
        :seed => seed,
        :sigma => sigma
    )
    return results
end