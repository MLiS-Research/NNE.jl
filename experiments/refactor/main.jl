"""
Main entry point for generating all publication plots.

This file includes all plotting modules and provides a single function
to generate all plots at once.

Usage:
    include("main.jl")
    generate_all_plots()
"""

# Include all plotting modules
include("plotting_utilities.jl")
include("exact_linear_plotting.jl")
include("tps_linear_plotting.jl")
include("linear_problem_plotting.jl")
include("tps_explanation_plotting.jl")

"""
    generate_all_plots(; seed=1141)

Generate all publication-quality plots and save them to the figures directory.

This function creates:
1. Exact linear perceptron plot (exact_linear_perceptron.pdf)
2. TPS linear perceptron plot (tps_linear_perceptron.pdf)
3. Comprehensive linear problem plot with 3 panels (all_linear_plots.pdf)
4. TPS perturbation examples (perturbation_examples.pdf)

# Arguments
- `seed`: Random seed for TPS perturbation examples (default: 1141)

# Returns
- Dictionary containing all generated figures
"""
function generate_all_plots(; seed=1141)
    println("\n" * "="^70)
    println("Generating all publication plots")
    println("="^70 * "\n")

    figures = Dict{String,Any}()

    # 1. Comprehensive linear problem plot
    println("1/2: Creating comprehensive linear problem plot...")
    figures["comprehensive_linear"] = save_comprehensive_linear_problem_figure()
    println("     ✓ Saved to figures/all_linear_plots.pdf\n")

    # 2. TPS perturbation examples
    println("2/2: Creating TPS perturbation examples...")
    figures["tps_perturbations"] = save_tps_perturbation_examples_figure(seed)
    println("     ✓ Saved to figures/perturbation_examples.pdf\n")

    println("="^70)
    println("All plots generated successfully!")
    println("="^70 * "\n")

    return figures
end

println("\nPlotting module loaded successfully!")
println("Run generate_all_plots() to create all figures")
