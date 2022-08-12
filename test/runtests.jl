using NNE
using NNE.Experimenter
using Test
using SafeTestsets

@safetestset "Experimenter" begin
    include("experimenter/experimenter.jl")
end