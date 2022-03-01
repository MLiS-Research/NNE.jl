function get_exponentially_spaced(min_val, max_val, num_points)
    return exp.((LinRange(log(min_val), log(max_val), num_points)))
end