using CairoMakie
using LaTeXStrings
using Measures: cm, mm, inch
using DataFrames

Makie = CairoMakie

default_dpi() = 144
default_fontsize() = 12

"""
    create_pub_fig(; dpi=default_dpi(), fontsize=default_fontsize(), num_panels=1, num_panels_y=1, kwargs...)

Create a publication-quality figure with specified DPI and font size.
The figure size is based on 8.6cm per panel width and a 21/28 aspect ratio per panel height.

# Arguments
- `dpi`: Dots per inch for the figure (default: 144)
- `fontsize`: Font size in points (default: 10)
- `num_panels`: Number of panels horizontally (default: 1)
- `num_panels_y`: Number of panels vertically (default: 1)
- `kwargs...`: Additional keyword arguments passed to Figure constructor

# Returns
- A CairoMakie Figure object configured for publication
"""
function create_pub_fig(; dpi=default_dpi(), fontsize=default_fontsize(), num_panels=1, num_panels_y=1, kwargs...)
    resolution = Int.(round.((8.6cm * num_panels, 8.6cm * 21 / 28 * num_panels_y) ./ (1inch) .* dpi))
    pt_in_mm = 0.352777777777778mm
    font_height = Int(round(fontsize * pt_in_mm / 1inch * dpi))
    f = Figure(; fontsize=font_height, fonts=(; regular="Computer Modern"), resolution, dpi, kwargs...)
    return f
end

"""
    get_marker_shape_dict(tau_values::AbstractArray{Int})
    get_marker_shape_dict(df::DataFrame)

Generate a mapping from tau values to marker shapes for plotting.

# Arguments
- `tau_values`: Array of integer tau values
- `df`: DataFrame containing a :tau column

# Returns
- Dictionary mapping tau values to marker symbols
"""
function get_marker_shape_dict(tau_values::AbstractArray{Int})
    possible_markers = [:circle, :diamond, :rect, :utriangle, :star4, :xcross]

    mapping = Dict{Int,Symbol}(
        t => m for (t, m) in Iterators.zip(tau_values, possible_markers)
    )
    return mapping
end

function get_marker_shape_dict(df::DataFrame)
    taus = sort(unique(df[!, :tau]))
    return get_marker_shape_dict(taus)
end
