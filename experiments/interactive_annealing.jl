using NNE.LinearTPSCalculation
using NNE
using TransitionPathSampling
using Plots
using LaTeXStrings
using ProgressBars
using Statistics
using Random
include("plotting/plotting_style.jl")

mutable struct TPSData{TProb,TAlg,TSol,TCache,TSVals,TEpoch}
    const problem::TProb
    const algorithm::TAlg
    const solution::TSol
    const cache::TCache
    const s_values::TSVals
    epoch::TEpoch
end

function init(τ, σ, s)
    problem = construct_problem(τ, σ)
    algorithm = construct_algorithm(τ, s, σ)

    solution = init_solution(algorithm, problem)
    cache = generate_cache(algorithm, problem)

    s_values = [(0, convert(Float64, s))]
    return TPSData(problem, algorithm, solution, cache, s_values, 0)
end

function step!(data::TPSData)
    data.epoch += 1
    step!(data.cache, data.solution, data.algorithm, data.epoch)
end

function step_many!(data::TPSData, num_epochs::Int)
    for _ in 1:num_epochs
        step!(data)
    end
end

function update_s!(data::TPSData, s::Float64)
    epoch = data.epoch
    push!(data.s_values, (epoch, s))
    data.algorithm.parameters.s = s
    nothing
end

function my_plot(losses, s_values; yerr=nothing, epoch_label_shift=2000, font_point_size=12, should_annotate=false, kwargs...)
    min_loss, max_loss = extrema(losses)
    colors = palette(:matter)
    diff(first.(s_values))
    epoch_widths = diff(first.(s_values))
    actual_s_values = last.(s_values)
    push!(epoch_widths, length(losses) - sum(epoch_widths) - 1)

    current_epoch = 1
    plt = nothing
    for (i, (s, w)) in enumerate(zip(actual_s_values, epoch_widths))
        plot_fn = (i == 1 ? plot : plot!)
        end_epoch = current_epoch + w
        c = colors[Int(round((i - 1) / (length(epoch_widths) - 1) * 255))+1]
        plt = plot_fn(current_epoch:end_epoch, view(losses, current_epoch:end_epoch); ribbon=(!isnothing(yerr) ? view(yerr, current_epoch:end_epoch) : nothing), yscale=:log10, c=c, label=LaTeXString("\$s=$(Int(round(s)))\$"))
        current_epoch += w
    end

    for (epoch, s) in s_values
        if epoch > 0
            vline!([epoch], c=:black, linestyle=:dash, label=nothing)
        end
        should_annotate && annotate!(epoch + epoch_label_shift, max_loss, Plots.text(LaTeXString("\$s=$s\$"), font_point_size, :black, :left))
    end
    xlabel!(plt, "TPS Epochs")
    ylabel!(plt, L"\mathbb{E} \left [ \ \overline{\mathcal{L}} \ \right ] / \tau")
    defaults = get_plot_defaults(; height_ratio=0.5)
    plot!(; legend=:topright, defaults..., kwargs...)
    return plt
end

function my_plot(data::TPSData; kwargs...)
    my_plot(data.solution.observations, data.s_values; kwargs...)
end

function my_plot(results::AbstractArray{TPSData}; kwargs...)
    mean_losses, err_mean_losses, s_values = process_results(results)
    my_plot(mean_losses, s_values; yerr=err_mean_losses, kwargs...)
end

function collect_all_data(; repeats=300, step_size=10000, num_steps=5, s_factor=2, s_initial=1.0, τ=8, σ=1.0, show_progress=false)
    results = Vector{TPSData}(undef, repeats)
    iter = show_progress ? ProgressBar(1:repeats) : (1:repeats)
    Threads.@threads for i in iter
        current_s = s_initial
        data = init(τ, σ, current_s)
        for j in 1:num_steps
            step_many!(data, step_size)
            if j < num_steps
                current_s *= s_factor
                update_s!(data, current_s)
            end
        end
        results[i] = data
    end

    return results
end

function process_results(results::AbstractArray{TPSData})
    losses = [d.solution.observations for d in results]
    s_values = first(results).s_values

    mean_losses = mean(losses)
    err_mean_losses = std(losses) / sqrt(length(results))

    return mean_losses, err_mean_losses, s_values
end

function run_experiment()
    Random.seed!(631398)
    step_size = 4000
    num_steps = 5
    results = collect_all_data(;
        repeats=100,
        step_size,
        num_steps,
        τ=8,
        σ=1.0,
        s_initial=1.0,
        s_factor=2
    )
    plt = my_plot(results)
    xtick_values = [step_size * i for i = 1:num_steps-1]
    xticks!(plt, xtick_values, string.(xtick_values))

    savefig(plt, "figures/linear_perceptron_annealed_loss.pdf")
    nothing
end