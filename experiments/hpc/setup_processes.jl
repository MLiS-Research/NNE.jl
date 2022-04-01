using Pkg
Pkg.activate(".")
using ArgParse
using Distributed
using ClusterManagers


function parse_commandline()
    arg_settings = ArgParseSettings()
    @add_arg_table arg_settings begin
        "--include_file"
            help = "The path to the file which should be included on each worker."
            arg_type = String
            default = nothing
        "--working_dir"
            help = "Path to the current working directory."
            arg_type = String
            default = nothing
        "--skip_activate_env"
            help = "Whether to skip activating the current environment on each worker."
            action = :store_true
        "--run_file"
            help = "An optional path to a file to run. This runs before the eval code."
            arg_type = String
            default = nothing
        "--eval_code", "-e"
            help = "Optional code to run."
            arg_type = String
            default = ""
    end

    parsed_args = parse_args(ARGS, arg_settings)
    return parsed_args
end



parsed_args = parse_commandline()
println("Parsed args:")
for (arg, val) in parsed_args
    if typeof(val) <: AbstractString
        println("   $arg => \"$val\"")
    else
        println("   $arg => $val")
    end
end

skip_activate = parsed_args["skip_activate_env"]
include_file = parsed_args["include_file"]
run_file = parsed_args["run_file"]
eval_code = parsed_args["eval_code"]
working_dir = parsed_args["working_dir"]

println("Setting up SLURM!")
# Setup SLURM
num_tasks = parse(Int, ENV["SLURM_NTASKS"])
cpus_per_task = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
ENV["JULIA_NUM_THREADS"] = cpus_per_task
addprocs(SlurmManager(num_tasks))

println("Workers: $(length(workers()))")

if !isnothing(working_dir)
    println("Switching to directory: $working_dir")
    eval(Meta.parse("@everywhere cd(\"$working_dir\");"))
end

if !skip_activate
    println("Activating environments at $(pwd())")
    @everywhere using Pkg;
    @everywhere Pkg.activate(".");
end

if !isnothing(include_file)
    include_file = abspath(include_file)
    println("Including $include_file")
    eval(Meta.parse("@everywhere include(\"$include_file\");"))
end

if !isnothing(run_file)
    run_file = abspath(run_file)
    println("Running file: $run_file")
    eval(Meta.parse("include(\"$run_file\");"))
end

if !isempty(eval_code)
    println("Running supplied code.")
    eval(Meta.parse(eval_code))
end

println("Finished!")

