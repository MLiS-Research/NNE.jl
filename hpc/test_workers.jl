using Distributed

function print_resources()
    return "Worker $(Distributed.myid()), Num Threads: $(Threads.nthreads())."
end