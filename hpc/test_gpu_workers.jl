using Distributed
using CUDA

function print_slurm_key!(io, key)
    if key in keys(ENV)
        println(io, "$key: $(ENV[key])")
    end
end

function print_resources()
    n = myid() * 1000

    # Allocate an array on the current, default GPU, to check memory
    CUDA.@sync arr = CUDA.zeros(Float32, n, n);

    io = IOBuffer()
    println(io, "Worker $(Distributed.myid())")
    println(io, "Num Threads: $(Threads.nthreads())")
    println(io, "CUDA Info:")
    CUDA.versioninfo(io)
    println(io, "Devices: $(collect(devices()))")
    println(io, "Hostname: $(gethostname())")
    println(io, "NVIDIA SMI:")
    println(io, read(`nvidia-smi`, String))
    print_slurm_key!(io, "SLURM_JOB_GPUS")
    print_slurm_key!(io, "SLURM_JOB_ID")
    print_slurm_key!(io, "SLURM_JOB_NODELIST")
    print_slurm_key!(io, "SLURM_TASK_PID")
    return String(take!(io))
end