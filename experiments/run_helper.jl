function get_exponentially_spaced(min_val, max_val, num_points)
    return exp.((LinRange(log(min_val), log(max_val), num_points)))
end


function get_git_hash()
    return strip(read(`git rev-parse HEAD`, String))
end