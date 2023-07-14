struct AlgorithmConfig{BT<:Union{Float32,Float64},ST<:Union{Float32,Float64}}
    bias::BT
    sigma::ST
    parameter_perturb_fraction::Float64
end
struct TensorboardLoggingConfig
    path::String
    train_loss_frequency::Int
    train_accuracy_frequency::Int
    validation_loss_frequency::Int
    validation_accuracy_frequency::Int
end
struct ExperimentConfig{AC<:AlgorithmConfig}
    trajectory_length::Int
    epochs::Int
    algorithm_config::AC
    use_progress::Bool
    tensorboard_logging_config::Union{Nothing,TensorboardLoggingConfig}
end