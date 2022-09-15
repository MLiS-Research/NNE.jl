include("tps_linear_plotting.jl")
include("../exact_linear_perceptron.jl")

function plot_all()
    s_values, t_values, _, _ = load_processed_tps_data()

    min_s = minimum(s_values)
    max_s = maximum(s_values)
    num_s_exact = 1000
    s_values_exact = exp_spaced_values(min_s, max_s, num_s_exact)

    unity_sigma_losses = reshape(calc_losses(s_values_exact, t_values, [1.0]), length(s_values_exact), length(t_values))
    unity_sigma_results = (s_values_exact, t_values, nothing, nothing, unity_sigma_losses)

    unity_sigma = construct_exact_linear_data_loss_vs_s_plot(false; results=unity_sigma_results)

    small_sigma_losses = reshape(calc_losses(s_values_exact, t_values, [0.1]), length(s_values_exact), length(t_values))
    small_sigma_results = (s_values_exact, t_values, nothing, nothing, small_sigma_losses)
    small_sigma_plt = construct_exact_linear_data_loss_vs_s_plot(false; results=small_sigma_results)

    empircal_plot = construct_tps_data_loss_vs_s_plot()

    full_width_defaults = get_plot_defaults_full_width()
    size = full_width_defaults[:size]
    dpi = full_width_defaults[:dpi]

    return plot(small_sigma_plt, unity_sigma, empircal_plot; layout=(1, 3), title=[L"(a)" L"(b)" L"(c)"], size, dpi)
end

function plot_all_and_save()
    plt = plot_all()

    savefig(plt, "figures/all_linear_plots.pdf")
end

