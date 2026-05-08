"""Render a standalone Bokeh preview for the current explorer API."""

from bokeh.io import output_file, save
from bokeh.layouts import column

from nuosclab import ExplorerConfig, NSIParams, compute_curves
from nuosclab.plotting import make_bokeh_probability_grid, make_bokeh_two_panel


def main() -> None:
    curves = compute_curves(
        ExplorerConfig(
            experiment="DUNE",
            nsi=NSIParams(eps_emu=0.04, delta_emu=0.6),
            n_points=200,
        )
    )
    output_file("bokeh-preview.html", title="nuosclab Bokeh preview")
    save(column(make_bokeh_two_panel(curves), make_bokeh_probability_grid(curves)))


if __name__ == "__main__":
    main()
