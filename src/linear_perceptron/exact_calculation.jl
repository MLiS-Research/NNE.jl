module ExactCalculation
using LinearAlgebra
using Random
using ForwardDiff
include("data_generation.jl")
calculate_γ(X) = X * transpose(X) ./ size(X, 2)
calculate_λ(X, y) = X * transpose(y) ./ size(X, 2)
calculate_C(y) = tr(y * transpose(y)) ./ size(y, 2)
alternative_loss(γ, λ, constant, W) = (0.5 * tr(W * γ * transpose(W)) - tr(W * λ) + constant)
get_identity_like(x) = Matrix{eltype(x)}(I, size(x)...)
function construct_Ã(s, τ, A, σ=1.0)
    len = size(A, 1)
    @assert size(A, 2) == len
    if τ == 1
        return s .* A
    end

    Ã = zeros(typeof(s), τ * len, τ * len)
    for t = 1:τ
        i = (t - 1) * len + 1
        multiplier = (t == 1 || t == τ) ? one(typeof(s)) : 2 * one(typeof(s))
        Ã[i:i+len-1, i:i+len-1] .= multiplier * I ./ (σ^2) + s .* A
    end
    id = -1.0 / (σ^2) .* get_identity_like(A)
    for t = 1:τ-1
        i = (t - 1) * len + 1
        Ã[i:i+len-1, i+len:i+2*len-1] .= id
        Ã[i+len:i+2*len-1, i:i+len-1] .= id
    end

    Ã
end
function construct_Ã_alt(s, τ, A, σ=1.0)
    len = size(A, 1)
    @assert size(A, 2) == len
    Ã = zeros(typeof(s), τ * len, τ * len)
    for t = 1:τ
        i = (t - 1) * len + 1
        multiplier = (t == 1 || t == τ) ? one(typeof(s)) : 2 * one(typeof(s))
        Ã[i:i+len-1, i:i+len-1] .= multiplier * ones(size(A)...) ./ (σ^2) + s .* A
    end
    id = -1.0 / (σ^2) .* ones(size(A)...)
    for t = 1:τ-1
        i = (t - 1) * len + 1
        Ã[i:i+len-1, i+len:i+2*len-1] .= id
        Ã[i+len:i+2*len-1, i:i+len-1] .= id
    end

    Ã
end
function construct_B̃(s, τ, B)
    sB = s .* B

    h, w = size(B)

    B̃ = zeros(typeof(s), τ * h, w)

    for t = 1:τ
        i = (t - 1) * h + 1
        B̃[i:i+h-1, :] .= sB
    end

    B̃
end

function log_z(s, τ, γ, λ, C, σ=1.0)
    Ã = construct_Ã(s, τ, γ, σ)
    B̃ = construct_B̃(s, τ, λ)
    detA = det(Ã)
    invA = inv(Ã)
    p = transpose(B̃) * invA * B̃
    return (tr(p) ./ 2 .- log(detA) ./ 2 .- (C / 2) * (τ) * s)
end

function log_z_separate(s, τ)
    Ã = construct_Ã(s, τ, γ)
    B̃ = construct_B̃(s, τ, λ)
    detA = det(Ã)
    invA = inv(Ã)
    p = transpose(B̃) * invA * B̃
    return [tr(p) ./ 2, 0 .- log(detA) ./ 2, 0 .- (C / 2) * (τ) * s]
end

function get_analytic_loss_from_ensemble(s, t, γ, λ, C, sigma)
    function get_log_z_value(s_factor)
        return log_z(s_factor, t, γ, λ, C, sigma)
    end
    grad_log_z = s_f -> ForwardDiff.derivative(get_log_z_value, s_f)
    grad_log = grad_log_z(s)
    k_val = -1.0 .* grad_log / (t - 1)
    return k_val
end

function calculate_analytical_loss(X, y)
    θ = (inv(X * X') * (X * y'))'
    l = loss(θ, X, y)
    return l
end


struct LinearProblem
    inputs
    outputs
    weights
    nsamples
    γ
    λ
    C
end

function construct_problem(seed=1234; noise=0.0)
    x, y, w = generate_seeded_data(; seed, noise)
    γ = calculate_γ(x)
    λ = calculate_λ(x, y)
    C = calculate_C(y)
    N = size(y)[end]
    return LinearProblem(x, y, w, N, γ, λ, C)
end

function get_mean_time_integrated_loss(s, τ, σ, problem::LinearProblem)
    closed_log_z_fn(s) = log_z(s, τ, problem.γ, problem.λ, problem.C, σ)
    grad_log_z_fn = s -> ForwardDiff.derivative(closed_log_z_fn, s)
    return -1.0 * grad_log_z_fn(s) / (τ)
end

export construct_problem, get_mean_time_integrated_loss

end


