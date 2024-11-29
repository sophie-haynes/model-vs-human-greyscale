import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from plotting_definition import plotting_definition_template


def run_plotting():
    
    plot_types = c.DEFAULT_PLOT_TYPES # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/simplified"
    # figure_dirname = "example-figures/test"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    run_plotting()
