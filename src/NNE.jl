module NNE

include("helpers/problem_generators.jl")
include("helpers/runner.jl")


include("linear_perceptron/exact_calculation.jl")
include("linear_perceptron/tps_calculation.jl")
include("classification/toy_problem.jl")


end