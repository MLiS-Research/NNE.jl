__precompile__()
module TensorboardLogger
import ...Logging: AbstractMetricsLogger, close, log_scalar_at
using PyCall
using Conda
mutable struct TBLogger <: AbstractMetricsLogger
    const path::String
    const file_writer::PyObject
    is_open::Bool
end

struct RealHyperparameter{T}
    name::Symbol
    min_value::T
    max_value::T
end
struct DiscreteHyperparameter{T}
    name::Symbol
    values::Vector{T}
end

function TBLogger(path::String)
    file_writer = py_tf.summary.create_file_writer(path)
    return TBLogger(path, file_writer, true)
end

function close(logger::TBLogger)
    logger.file_writer.close()
    logger.is_open = false
end

function log_scalar_at(logger::TBLogger, key::Symbol, data, epoch::Int)
    if logger.is_open

    else
        error("Tried to write to a closed logger. Logger path $(logger.path).")
    end
end

convert_to_py_hparam(hp::RealHyperparameter) = py_create_real_hparam(string(hp.name), hp.min_value, hp.max_value)
convert_to_py_hparam(hp::DiscreteHyperparameter) = py_create_discrete_hparam(string(hp.name), hp.values)

function setup_hparams!(logger::TBLogger, config::AbstractArray{Union{RealHyperparameter,DiscreteHyperparameter}}, metrics::Dict{Symbol,String})
    hparams = map(convert_to_py_hparam, config)
    metrics = [py_create_metric(string(key), display_name) for (key, display_name) in metrics]

    py_setup_hparams(logger, hparams, metrics)
    nothing
end
function log_hparams!(logger::TBLogger, config::Dict{Symbol,Any})
    converted_config = Dict((string(k) => v for (k, v) in config)...)
    py_log_hparams(logger, converted_config)
    nothing
end



const py_tf = PyNULL()
const py_tensorboard = PyNULL()
const py_log_scalar = PyNULL()
const py_create_discrete_hparam = PyNULL()
const py_create_real_hparam = PyNULL()
const py_create_metric = PyNULL()
const py_setup_hparams = PyNULL()
const py_log_hparams = PyNULL()

function __init__()
    # Setup
    Conda.pip_interop(true)
    Conda.pip("install", "tensorflow")
    Conda.pip("install", "tensorboard")

    copy!(py_tf, pyimport("tensorflow"))
    copy!(py_tensorboard, pyimport("tensorboard"))


    py"""
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp
    import tensorboard
    def log_scalar(logger, key, data, epoch):
        with logger.as_default():
            tf.summary.scalar(key, data, step=epoch)
    def create_discrete_hparam(key, allowed_values):
        return hp.HParam(key, hp.Discrete(allowed_values))
    def create_real_interval_hparam(key, minvalue, maxvalue):
        return hp.HParam(key, hp.RealInterval(minvalue, maxvalue))
    def create_metric(key, display_name):
        return hp.Metric(key, display_name)
    def setup_hparams(logger, hparams, metrics):
        with logger.as_default():
            hp.hparams_config(hparams=hparams, metrics=metrics)
    def log_hparams(logger, config):
        with logger.as_default():
            hp.hparams(config)
    """

    copy!(py_create_discrete_hparam, py"create_discrete_hparam")
    copy!(py_create_real_hparam, py"create_real_interval_hparam")
    copy!(py_create_metric, py"create_metric")
    copy!(py_setup_hparams, py"setup_hparams")
    copy!(py_log_hparams, py"log_hparams")
end

export TBLogger, RealHyperparameter, DiscreteHyperparameter, setup_hparams!, log_hparams!, log_scalar_at, close
end