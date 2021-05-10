## PLOTTING
import os

import colorcet as cc
from numpy import linspace
from scipy.stats.kde import gaussian_kde

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, Legend
from bokeh.plotting import figure
from bokeh.sampledata.perceptions import probly
from bokeh.models import HBar

from bokeh.layouts import layout

import numpy as np


def multi_feature_plotter(feature_extractors_list, title_prefix="ridgeplot", force_extract=False,
                      plot_width=800, plot_height=1200, legend_fnt_size="8px",
                      scale_y=True, resolution=1000, plot_with_complement=False):

    # Extract features if not done already or force_extract required
    # also get the major/minor features available in the feature set
    # exp:  feat_major_fields   ["statistical", "micro-timing"]
    #       feat_minor_fields   ["noi", "lowness", "swing"]

    feat_major_fields = list()
    feat_minor_fields = list()

    for i, feature_extractor in enumerate(feature_extractors_list):
        feature_extractor.extract(force_extract=force_extract)
        majors, minors = feature_extractor.get_major_minor_field_keys()
        feat_major_fields.extend(majors)
        feat_minor_fields.extend(minors)

    feat_major_fields = list(set(feat_major_fields))
    feat_minor_fields = list(set(feat_minor_fields))

    number_of_sets = len(feature_extractors_list)  # Number of feature extractors

    figure_layout = []

    # Each bokeh figure will show a single features value across all sets (accessible via feature_extractors_list)
    for maj_key in feat_major_fields:

        for minor_key in feat_minor_fields:
            data_list = list()
            tags = list()
            for i in range(number_of_sets):
                if minor_key in feature_extractors_list[i].extracted_features[maj_key].keys():
                    data_for_feature = feature_extractors_list[i].extracted_features[maj_key][minor_key]
                    data_for_feature = np.where(np.isnan(data_for_feature), 0, data_for_feature)
                    data_for_feature = np.where(np.isinf(data_for_feature), 0, data_for_feature)
                    data_list.append(data_for_feature)
                    tags.append("{} ".format(feature_extractors_list[i].name))
                    title = "{}::{}::{}".format(title_prefix, maj_key, minor_key)
            if len(data_list) > 1:
                if plot_with_complement is True:
                    figure_layout.append(
                        ridge_kde_multi_feature_with_complement_set(
                            tags, data_list,
                            title=title,
                            resolution=resolution, plot_width=plot_width, plot_height=plot_height,
                            legend_fnt_size=legend_fnt_size, scale_y=scale_y)
                    )
                else:
                    figure_layout.append(
                        ridge_kde_multi_feature(
                            tags, data_list, resolution=resolution,
                            plot_width=plot_width, plot_height=plot_height,
                            title=title,
                            legend_fnt_size=legend_fnt_size,
                            plot_set_complement=False, scale_y=scale_y)
                    )

    return figure_layout

def ridge_kde_multi_feature_with_complement_set(tags, data_list,
                                            title="",
                                            resolution=1000,
                                            plot_width=1200, plot_height=1200,
                                            legend_fnt_size="8px",
                                            scale_y=True):

    p1 = ridge_kde_multi_feature(tags, data_list, title=title,
                             resolution=resolution,
                             plot_width=plot_width, plot_height=plot_height,
                             legend_fnt_size=legend_fnt_size,
                             plot_set_complement=False, scale_y=scale_y)

    p2 = ridge_kde_multi_feature(tags, data_list, title=title,
                             resolution=resolution,
                             plot_width=plot_width, plot_height=plot_height,
                             legend_fnt_size=legend_fnt_size,
                             plot_set_complement=True, scale_y=scale_y)

    p2.x_range = p1.x_range
    p2.y_range = p1.y_range

    return [p1, p2]


def ridge_kde_multi_feature(tags, data_list, title="",
                        resolution=1000,
                        plot_width=1200, plot_height=2000,
                        legend_fnt_size="8px",
                        plot_set_complement=False, scale_y=True):

    #todo add median, and quartiles to plot

    # create color palette (max index of cc.rainbow is 256)
    n_sets = len(data_list)
    palette_resolution = int(254//n_sets)
    palette = [cc.rainbow[i*palette_resolution] for i in range(n_sets)]

    # Find range of data
    # Find range of data
    data_range_min = []
    data_range_max = []

    for ix, data in enumerate(data_list):
        if data.size < 2:
            data = np.zeros(100)
            data_list[ix] = data
        data_range_min = np.floor(min([min(data)]) - 1)
        data_range_max = np.ceil(max([max(data)]) + 1)

    # create x axis data with required resolution
    x = linspace(data_range_min, data_range_max, resolution)
    # Create bokeh ColumnDataSource containing data structure
    source = ColumnDataSource(data=dict(x=x))

    def ridge(category, data):
        scale = (1/data.max()) if scale_y is True else 1
        return list(zip([category]*len(data), scale*data))

    p = figure(y_range=tags, plot_width=plot_width, plot_height=plot_height, x_range=(data_range_min, data_range_max))

    legend_it = []
    for ix, data in enumerate(data_list):
        if plot_set_complement is True:
            complementary_data = np.array([])
            for ix_ in range(n_sets):
                if ix_ != ix:
                    complementary_data = np.append(complementary_data, data_list[ix_].flatten())
            data = complementary_data

        # Find kernel bandwidth using Scott's Rule of Thumb
        # https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
        tag = tags[ix]
        if data.mean() == 0.0 and data.std() == 0.0:
            y = ridge(tags[ix], data)
        else:
            pdf = gaussian_kde(data)
            y = ridge(tags[ix], pdf(x))
        source.add(y, tag)
        legend_ = "{} ~ N({}, {})".format(tags[ix], round(data.mean(), 2), round(data.std(), 2))
        legend_ = ("~"+legend_ ) if plot_set_complement else legend_
        c = p.patch('x', tags[ix], alpha=0.6, color=palette[ix], line_color="black", source=source)
        legend_it.append((legend_, [c]))

        #p.patch('x', tags[ix], alpha=0.6, color=palette[ix], line_color="black", source=source,
        #        legend=legend_)
        #p.legend.location = "top_right"
        #p.legend.click_policy = "hide"
        #p.legend.label_text_font_size = legend_fnt_size
        p.title=title

    legend = Legend(items=legend_it)
    legend.label_text_font_size = legend_fnt_size
    legend.click_policy = "hide"
    p.add_layout(legend, 'right')

    p.outline_line_color = None
    p.background_fill_color = "#efefef"

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#dddddd"
    p.xgrid.ticker = p.xaxis.ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None

    p.y_range.range_padding = 0.12

    return p







if __name__ == '__main__':

    output_file("ridgeplot.html")
    set_a = np.random.rand(200)
    set_b = 3*np.random.rand(100)-1
    set_c = 3*np.random.rand(100)+4
    """
    p = ridge_kde_multi_feature(tags=["rand1", "rand2", "rand3"], data_list=[set_a, set_b, set_b],
                            plot_set_complement=True)
    show(p)
    output_file("ridgeplot2.html")
    p = ridge_kde_multi_feature(tags=["rand1", "rand2", "rand3"], data_list=[set_a, set_b, set_b],
                            plot_set_complement=False)
    show(p)"""

    output_file("combined.html")

    p = ridge_kde_multi_feature_with_complement_set(tags=["rand1", "rand2", "rand3"], data_list=[set_a, set_b, set_b],
                                                resolution=1000,
                                                plot_width=600, plot_height=1200, legend_fnt_size="8px",
                                                scale_y=True)

    show(layout(p))
