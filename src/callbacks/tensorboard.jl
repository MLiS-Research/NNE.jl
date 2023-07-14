module Tensorboard

using TensorBoardLogger, Logging
using SimpleTraits
import TransitionPathSampling.Callbacks as CB

struct TBLoggerCallback{T} <: CB.AbstractCallback
    logger::TBLogger
    metric_gatherers::T
end

function TBLoggerCallback(path, metrics...; conflict_option=tb_append)
    logger = TBLogger(path, conflict_option)
    return TBLoggerCallback(logger, Tuple(m for m in metrics))
end

abstract type AbstractMetricGatherer end
gather(gatherer::AbstractMetricGatherer, deps::CB.SolveDependencies) = error("Unimplemented")
tag(gatherer::AbstractMetricGatherer) = error("Unimplemented")
frequency(gatherer::AbstractMetricGatherer) = 1


function log_metric!(logger::TBLogger, gatherer::AbstractMetricGatherer, deps::CB.SolveDependencies)
    TensorBoardLogger.log_value(logger, tag(gatherer), gather(gatherer, deps))
    nothing
end

@traitimpl CB.RunsPostInnerLoop{TBLoggerCallback}
function CB.run(cb::TBLoggerCallback, deps::CB.SolveDependencies)
    current_epoch = Int(deps.iterator_state)
    for gatherer in cb.metric_gatherers
        if current_epoch % frequency(gatherer) # throttle logging
            log_metric!(cb.logger, gatherer, deps)
        end
    end
end

export TBLoggerCallback

end