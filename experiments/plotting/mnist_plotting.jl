using MLDatasets
using Plots
using Plots.PlotMeasures
using Images
using Flux
using Statistics
using DataFrames
using NNE.MNISTTraining
using NNE.Experimenter
using Base.Iterators
using BSON: @save, @load
include("plotting_style.jl")

function get_examples(digits...)
    labels = MNIST.trainlabels(1:100)
    indices = [findlast(labels .== d) for d in digits]
    data = MNIST.traintensor(Float32, indices)
    image_data = [Gray.(reshape(data[:, :, i], 28, 28)') for i = 1:length(indices)]
    return image_data
end

function create_image_plots(digits...)
    defaults = get_plot_defaults()

    letters = ["($c)" for c in ('a':'z')[1:length(digits)]]
    images = get_examples(digits...)
    plts = []
    for (img, l) in zip(images, letters)
        plt = plot(img; ticks=false, title=l, titleloc=:left, defaults)
        push!(plts, plt)
    end

    plt = plot(plts...)
    return plt
end

function reconstruct_mnist_models(info_dict; outputs=2, device=cpu)
    model = generate_mnist_model(; outputs)
    _, re = Flux.destructure(model)
    if info_dict[:τ] == 1
        model = re(info_dict[:final_state]) # reconstruct model from params
        return [model |> device]
    end

    models = [re(state) |> device for state in info_dict[:final_state]]
    return models
end

function measure_train_accuracy(info_dict; device=cpu, outputs=2)
    models = reconstruct_mnist_models(info_dict; outputs, device)
    features = info_dict[:features] |> device
    labels = info_dict[:labels] |> cpu
    accuracies = [Flux.mean(reshape((x -> x[1] - 1).(argmax(m(features) |> cpu, dims=1)), :) .== labels) for m in models]
    return accuracies
end

function measure_test_accuracy(info_dict; device=cpu, outputs=2)
    models = reconstruct_mnist_models(info_dict; outputs, device)
    features, labels = get_mnist_testing_dataset(; device, outputs)
    labels = labels |> cpu
    accuracies = [Flux.mean(reshape((x -> x[1] - 1).(argmax(m(features) |> cpu, dims=1)), :) .== labels) for m in models]
    return accuracies
end

function plot_s_vs_loss(trials::AbstractArray{Trial}; kwargs...)
    df = DataFrame(trials)
    prepare_trials_df!(df)
    plot_s_vs_loss(df; kwargs...)
end
function plot_s_vs_loss(df::DataFrame; max_loss_samples=typemax(Int), kwargs...)
    trajectory_lengths = sort(collect(Set(df.τ)))
    defaults = get_plot_defaults();
    plt = plot(; )
    marker_shapes = (:circle, :rect, :dtriangle, :utriangle, :diamond, :pentagon)
    for (i, t) in enumerate(trajectory_lengths)
        sub_df = df[df.τ .== t, :]
        s_vals = sort(collect(Set(sub_df.s)))
        losses = Float64[]
        errors = Float64[]
        for s in s_vals
            repeat_ls = (x-> length(x) > max_loss_samples ? x[end-max_loss_samples+1:end] : x).(sub_df[sub_df.s .== s, :losses])
            num_repeats = length(repeat_ls)
            push!(losses, mean(mean(ls) for ls in repeat_ls)/t)
            # errors_repeats = [std(ls)/sqrt(length(ls)) for ls in sub_df[sub_df.s .== s, :losses]]
            # total_error = sqrt(sum(x->x*x, errors_repeats))/num_repeats/t
            total_error = std(mean(ls) for ls in repeat_ls)/t/sqrt(num_repeats)
            push!(errors, total_error)
        end
        scatter!(plt, s_vals, losses; label="τ=$t", defaults..., yerror=errors, markershape=marker_shapes[(i-1)%length(marker_shapes)+1], kwargs...)
    end
    plot!(plt; xscale=:log10, yscale=:log10, legend=:bottomleft, defaults...)
    xlabel!(plt, "s")
    ylabel!(plt, "<L>/τ")
    return plt
end
function plot_accuracy_vs_loss(trials::AbstractArray{Trial}; device=gpu, outputs=10, kwargs...)
    df = DataFrame(trials)
    prepare_trials_df!(df)
    trajectory_lengths = sort(collect(Set(df.τ)))
    defaults = get_plot_defaults();
    plt = plot(; )
    marker_shapes = (:circle, :rect, :dtriangle, :utriangle, :diamond, :pentagon)
    for (i, t) in enumerate(trajectory_lengths)
        sub_df = df[df.τ .== t, :]
        s_vals = sort(collect(Set(sub_df.s)))
        accuracies = Float64[]
        errors = Float64[]
        for s in s_vals
            results_list = [x.results for x in trials if x.results[:τ] == t && x.results[:s] == s]
            repeat_accs = measure_train_accuracy.(results_list; device, outputs)
            num_repeats = length(repeat_accs)
            push!(accuracies, mean(mean(as) for as in repeat_accs))
            push!(errors, std(mean(as) for as in repeat_accs)/sqrt(num_repeats))
        end
        scatter!(plt, s_vals, accuracies .* 100; label="τ=$t", defaults..., yerror=errors, markershape=marker_shapes[(i-1)%length(marker_shapes)+1], kwargs...)
    end
    plot!(plt; xscale=:log10, yscale=:log10, legend=:topleft, ylims=(8, 100), defaults...)
    xlabel!(plt, "s")
    ylabel!(plt, "Train Accuracy (%)")
    return plt
end
function plot_s_vs_acceptance(trials::AbstractArray{Trial}; kwargs...)
    df = DataFrame(trials)
    prepare_trials_df!(df)
    plot_s_vs_acceptance(df; kwargs...)
end
function plot_s_vs_acceptance(df::DataFrame; max_acceptance_samples=typemax(Int), kwargs...)
    trajectory_lengths = sort(collect(Set(df.τ)))
    plt = plot(;)
    marker_shapes = (:circle, :rect, :dtriangle, :utriangle, :diamond)
    for (i, t) in enumerate(trajectory_lengths)
        sub_df = df[df.τ .== t, :]
        s_vals = sort(collect(Set(sub_df.s)))
        acceptances = Float64[]
        errors = Float64[]
        for s in s_vals
            repeat_accepts = (x-> length(x) > max_acceptance_samples ? x[end-max_acceptance_samples+1:end] : x).(sub_df[sub_df.s .== s, :acceptances])
            num_repeats = length(repeat_accepts)
            push!(acceptances, mean(mean(as) for as in repeat_accepts))
            total_error = std(mean(as) for as in repeat_accepts)/sqrt(num_repeats)
            push!(errors, total_error)
        end
        scatter!(plt, s_vals, acceptances; label="τ=$t", yerror=errors, markershape=marker_shapes[(i-1)%length(marker_shapes)+1], kwargs...)
    end
    plot!(plt; xscale=:log10, yscale=:log10)
    xlabel!(plt, "s")
    ylabel!(plt, "<A>")
    return plt
end
function prepare_trials_df!(trials_df::DataFrame)
    insertcols!(trials_df, :losses => (x->x[:observations]).(trials_df.results))
    insertcols!(trials_df, :acceptances => (x->Float64.(diff(x[:observations]).==0)).(trials_df.results))
    insertcols!(trials_df, :s => (x->x[:s]).(trials_df.configuration))
    insertcols!(trials_df, :τ => (x->x[:τ]).(trials_df.configuration))
    insertcols!(trials_df, :σ => (x->x[:σ]).(trials_df.configuration))
end

function plot_avg_loss(results, new_plot=true; should_scale_x=false, kwargs...)
    max_len = minimum([length(x[:observations]) for x in results])
    losses = (x->x[:observations][1:max_len]).(results)
    med_duration = median((x->x[:duration].value).(results)) ./ 1000.0
    tau = mean((x->x[:τ]).(results))
    mean_loss = mean(losses) / tau
    std_loss = std(losses) / sqrt(length(losses)) / tau
    plot_fn = new_plot ? plot : plot!
    x_scale = should_scale_x ? LinRange(0, med_duration, length(mean_loss)) : 1:length(mean_loss)
    plt = nothing
    if length(mean_loss) > 5e5
        mean_loss = conv_1d(mean_loss, 100, 100.0)
        std_loss = conv_1d(std_loss, 100, 100.0)
        plt = @views plot_fn(x_scale[begin:1000:end], mean_loss[begin:1000:end]; ribbon=(std_loss[begin:1000:end], std_loss[begin:1000:end]), legend=false, kwargs...)
    else
        plt = plot_fn(x_scale, mean_loss; ribbon=(std_loss, std_loss), legend=false, kwargs...)
    end
    xlabel!(should_scale_x ? "Runtime (s)" : "Epochs")
    ylabel!("Mean Loss")
    return plt
end

function plot_avg_loss_compared(trials; max_y_lim=nothing, kwargs...)
    ts = sort(collect(Set(x.configuration[:τ] for x in trials)))
    ss = sort(collect(Set(x.configuration[:s] for x in trials)))
    split_trials = [[tr for tr in trials if tr.configuration[:τ]==t && tr.configuration[:s]==s] for (s, t) in product(ss, ts)]


    plts = []
    for (j, s) in enumerate(ss)
        plt = nothing
        for (i, t) in enumerate(ts)
            plt = plot_avg_loss([trial.results for trial in split_trials[j, i]], (i==1); label="τ=$t", legend=:outerright, kwargs...)
            title!(plt, "($(Char(96+j)))")
            if !isnothing(max_y_lim)
                ylims!(plt, 0, max_y_lim)
            end
        end
        plot!(plt; titlelocation=:left)
        push!(plts, plt)
    end
    return plot(plts...; layout=(length(plts), 1), dpi=300, size=(600, 1200), left_margin = [10mm 0mm])
end

function conv_1d(y, w=500, sigma=100.0)
    f(x) = exp(-0.5 * x * x / (sigma * sigma)) / (sigma * sqrt(2*π))
    kernel = f.(collect(-w:w))
    kernel = kernel ./ sum(kernel)
    conv_y = similar(y)
    for i in eachindex(y)
        min_i = max(1, i-w)
        max_i = min(length(y), i+w)
        kernel_r = (w-(i-min_i)+1):(w+(max_i-i)+1)
        r = (min_i:max_i)
        norm_kernel = kernel[kernel_r]
        norm_kernel ./= sum(norm_kernel)
        conv_y[r] .= sum(y[r].*norm_kernel)
    end
    return conv_y
end

function plot_acceptance(results, new_plot=true; should_scale_x=false, kwargs...)
    acceptances = (x->Float64.(x[:acceptances])).(results)
    conv_acceptances = (x->conv_1d(x)).(acceptances)
    med_duration = median((x->x[:duration].value).(results)) ./ 1000.0
    mean_acceptances = mean(conv_acceptances)
    std_acceptances = std(conv_acceptances)
    plot_fn = new_plot ? plot : plot!
    x_scale = should_scale_x ? LinRange(0, med_duration, length(mean_acceptances)) : 1:length(mean_acceptances)
    plt = plot_fn(x_scale, mean_acceptances; ribbon=(std_acceptances, std_acceptances), legend=false, kwargs...)
    xlabel!(should_scale_x ? "Runtime (s)" : "Epochs")
    ylabel!("Mean Acceptance")
    return plt
end


function prepare_mnist_results(trials::AbstractArray{Trial}; max_loss_samples=typemax(Int), device=gpu, outputs=10, kwargs...)
    trajectory_lengths = sort(collect(Set([x.configuration[:τ] for x in trials])))

    results = Dict{Symbol, Any}()
    results[:trajectory_lengths] = trajectory_lengths
    results[:data] = Dict{Int, Any}()

    for (i, t) in enumerate(trajectory_lengths)
        s_vals = sort(collect(Set([x.configuration[:s] for x in trials if x.configuration[:τ]==t])))

        t_data = Dict{Float64, Any}()
        for (j, s) in enumerate(s_vals)
            s_data = Dict{Symbol, Any}()
            loss_arrays = [x.results[:observations] for x in trials if x.configuration[:τ]==t && x.configuration[:s] == s]
            repeat_ls = (x-> length(x) > max_loss_samples ? x[end-max_loss_samples+1:end] : x).(loss_arrays)
            num_repeats = length(repeat_ls)
            s_data[:loss] = mean(mean(ls) for ls in repeat_ls)/t
            errors_repeats = [std(ls)/sqrt(length(ls)) for ls in loss_arrays]
            total_error = sqrt(sum(x->x*x, errors_repeats))/num_repeats/t
            s_data[:loss_error] = total_error

            
            results_list = [x.results for x in trials if x.results[:τ] == t && x.results[:s] == s]
            repeat_accs = measure_train_accuracy.(results_list; device, outputs)
            num_repeats = length(repeat_accs)
            s_data[:accuracy] = mean(mean(as) for as in repeat_accs)
            s_data[:accuracy_error] = std(mean(as) for as in repeat_accs)/sqrt(num_repeats)
            s_data[:s] = s
            t_data[s] = s_data
        end
        results[:data][t] = t_data
    end

    @save get_mnist_results_save_path() results
    nothing
end

get_mnist_results_save_path() = joinpath("results", "mnist_data.bson")
get_mnist_results_figure_s_save_path() = joinpath("figures", "full_mnist_s_ensemble.pdf")
get_mnist_results_figure_accuracy_save_path() = joinpath("figures", "full_mnist_accuracy.pdf")

function get_mnist_results()
    results = nothing
    @load get_mnist_results_save_path() results
    return results
end

function plot_mnist_s_graph()
    results = get_mnist_results()

    t_values = sort(results[:trajectory_lengths])
    s_values = sort(collect(Set(vcat([collect(keys(d)) for d in values(results[:data])]...))))

    losses = zeros(Float64, length(s_values), length(t_values))
    errors = similar(losses)
    for (i, (s, t)) in enumerate(Base.product(s_values, t_values))
        losses[i] = results[:data][t][s][:loss]
        errors[i] = results[:data][t][s][:loss_error]
    end

    marker_shapes = (:circle, :rect, :dtriangle, :utriangle, :diamond, :star5)
    wrapped_shapes = reshape([marker_shapes[(i-1)%length(marker_shapes)+1] for i in 1:length(t_values)], 1, :)
    plt = plot_s_graph(s_values, t_values, losses; linecolor=nothing, markershape=wrapped_shapes, legend=:bottomleft)

    return plt
end

function plot_mnist_accuracy_graph()
    results = get_mnist_results()

    t_values = sort(results[:trajectory_lengths])
    s_values = sort(collect(Set(vcat([collect(keys(d)) for d in values(results[:data])]...))))

    accuracies = zeros(Float64, length(s_values), length(t_values))
    accuracy_errors = similar(accuracies)
    for (i, (s, t)) in enumerate(Base.product(s_values, t_values))
        accuracies[i] = results[:data][t][s][:accuracy] * 100
        accuracy_errors[i] = results[:data][t][s][:accuracy_error] * 100
    end

    marker_shapes = (:circle, :rect, :dtriangle, :utriangle, :diamond, :star5)
    wrapped_shapes = reshape([marker_shapes[(i-1)%length(marker_shapes)+1] for i in 1:length(t_values)], 1, :)
    plt = plot_s_graph(s_values, t_values, accuracies; linecolor=nothing, markershape=wrapped_shapes, legend=:right, yerr=accuracy_errors)
    ylabel!(plt, "Accuracy (%)")
    ylims!(plt, 8, 100)
    return plt
end


function plot_and_save_mnist_s_graph()
    plt = plot_mnist_s_graph()

    savefig(plt, get_mnist_results_figure_s_save_path())
    nothing
end


function plot_and_save_mnist_accuracy_graph()
    plt = plot_mnist_accuracy_graph()

    savefig(plt, get_mnist_results_figure_accuracy_save_path())
    nothing
end