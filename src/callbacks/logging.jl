module Logging

unimplemented() = error("Unimplemented")
abstract type AbstractMetricsLogger end
close(logger::AbstractMetricsLogger) = unimplemented()
log_scalar_at(logger::AbstractMetricsLogger, key::Symbol, data, epoch::Int) = unimplemented()

include("backends/tensorboard.jl")

end