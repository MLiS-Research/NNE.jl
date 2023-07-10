struct ExperimentConfig{AC<:AlgorithmConfiguration}
    trajectory_length::Int
    epochs::Int
    algorithm_config::AC
    use_progress::Bool
end
struct AlgorithmConfig{BT<:Union{Float32,Float64},ST<:Union{Float32,Float64}}
    bias::BT
    sigma::ST
    parameter_perturb_fraction::Float64
end
