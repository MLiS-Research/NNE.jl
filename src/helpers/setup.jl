using Random

function generate_trajectory(initial_state, τ, σ; rng=Random.GLOBAL_RNG)
    states = [similar(initial_state) for _ = 1:τ]
    states[begin] .= initial_state

    for t = 2:τ
        randn!(rng, states[t])

        states[t] .= states[t] .* σ .+ states[t-1]
    end

    return states
end