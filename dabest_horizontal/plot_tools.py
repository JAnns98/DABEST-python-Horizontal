#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com
# A set of convenience functions used for producing plots in `dabest`.


from .misc_tools import merge_two_dicts
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

def halfviolin(v, half='right', fill_color='k', alpha=1,
                line_color='k', line_width=0):
    import numpy as np

    for b in v['bodies']:
        V = b.get_paths()[0].vertices

        mean_vertical = np.mean(V[:, 0])
        mean_horizontal = np.mean(V[:, 1])

        if half == 'right':
            V[:, 0] = np.clip(V[:, 0], mean_vertical, np.inf)
        elif half == 'left':
            V[:, 0] = np.clip(V[:, 0], -np.inf, mean_vertical)
        elif half == 'bottom':
            V[:, 1] = np.clip(V[:, 1], -np.inf, mean_horizontal)
        elif half == 'top':
            V[:, 1] = np.clip(V[:, 1], mean_horizontal, np.inf)

        b.set_color(fill_color)
        b.set_alpha(alpha)
        b.set_edgecolor(line_color)
        b.set_linewidth(line_width)



# def align_yaxis(ax1, v1, ax2, v2):
#     """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
#     # Taken from
#     # http://stackoverflow.com/questions/7630778/
#     # matplotlib-align-origin-of-right-axis-with-specific-left-axis-value
#     _, y1 = ax1.transData.transform((0, v1))
#     _, y2 = ax2.transData.transform((0, v2))
#     inv = ax2.transData.inverted()
#     _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
#     miny, maxy = ax2.get_ylim()
#     ax2.set_ylim(miny+dy, maxy+dy)
#
#
#
# def rotate_ticks(axes, angle=45, alignment='right'):
#     for tick in axes.get_xticklabels():
#         tick.set_rotation(angle)
#         tick.set_horizontalalignment(alignment)



def get_swarm_spans(coll):
    """
    Given a matplotlib Collection, will obtain the x and y spans
    for the collection. Will return None if this fails.
    """
    import numpy as np
    x, y = np.array(coll.get_offsets()).T
    try:
        return x.min(), x.max(), y.min(), y.max()
    except ValueError:
        return None



def error_bar(data, x, y, type='mean_sd', offset=0.2, ax=None,
              line_color="black", gap_width_percent=1, pos=[0, 1],
              method='gapped_lines', **kwargs):
    '''
    Function to plot the standard deviations as vertical errorbars.
    The mean is a gap defined by negative space.

    This function combines the functionality of gapped_lines(),
    proportional_error_bar(), and sankey_error_bar().

    Keywords
    --------
    data: pandas DataFrame.
        This DataFrame should be in 'long' format.

    x, y: string.
        x and y columns to be plotted.

    type: ['mean_sd', 'median_quartiles'], default 'mean_sd'
        Plots the summary statistics for each group. If 'mean_sd', then the
        mean and standard deviation of each group is plotted as a gapped line.
        If 'median_quantiles', then the median and 25th and 75th percentiles of
        each group is plotted instead.

    offset: float (default 0.3) or iterable.
        Give a single float (that will be used as the x-offset of all
        gapped lines), or an iterable containing the list of x-offsets.

    line_color: string (matplotlib color, default "black") or iterable of
        matplotlib colors.

        The color of the vertical line indicating the standard deviations.

    gap_width_percent: float, default 5
        The width of the gap in the line (indicating the central measure),
        expressed as a percentage of the y-span of the axes.

    ax: matplotlib Axes object, default None
        If a matplotlib Axes object is specified, the gapped lines will be
        plotted in order on this axes. If None, the current axes (plt.gca())
        is used.

    pos: list, default [0, 1]
        The positions of the error bars for the sankey_error_bar method.

    method: string, default 'gapped_lines'
        The method to use for drawing the error bars. Options are:
        'gapped_lines', 'proportional_error_bar', and 'sankey_error_bar'.

    kwargs: dict, default None
        Dictionary with kwargs passed to matplotlib.lines.Line2D
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    if gap_width_percent < 0 or gap_width_percent > 100:
        raise ValueError("`gap_width_percent` must be between 0 and 100.")
    if method not in ['gapped_lines', 'proportional_error_bar', 'sankey_error_bar']:
        raise ValueError("Invalid `method`. Must be one of 'gapped_lines', 'proportional_error_bar', or 'sankey_error_bar'.")

    if ax is None:
        ax = plt.gca()
    ax_ylims = ax.get_ylim()
    ax_yspan = np.abs(ax_ylims[1] - ax_ylims[0])
    gap_width = ax_yspan * gap_width_percent / 100

    keys = kwargs.keys()
    if 'clip_on' not in keys:
        kwargs['clip_on'] = False

    if 'zorder' not in keys:
        kwargs['zorder'] = 5

    if 'lw' not in keys:
        kwargs['lw'] = 2.

    if isinstance(data[x].dtype, pd.CategoricalDtype):
        group_order = pd.unique(data[x]).categories
    else:
        group_order = pd.unique(data[x])

    means = data.groupby(x)[y].mean().reindex(index=group_order)

    if method in ['proportional_error_bar', 'sankey_error_bar']:
        g = lambda x: np.sqrt((np.sum(x) * (len(x) - np.sum(x))) / (len(x) * len(x) * len(x)))
        sd = data.groupby(x)[y].apply(g)
    else:
        sd = data.groupby(x)[y].std().reindex(index=group_order)

    lower_sd = means - sd
    upper_sd = means + sd

    if (lower_sd < ax_ylims[0]).any() or (upper_sd > ax_ylims[1]).any():
        kwargs['clip_on'] = True

    medians = data.groupby(x)[y].median().reindex(index=group_order)
    quantiles = data.groupby(x)[y].quantile([0.25, 0.75]) \
        .unstack() \
        .reindex(index=group_order)
    lower_quartiles = quantiles[0.25]
    upper_quartiles = quantiles[0.75]

    if type == 'mean_sd':
        central_measures = means
        lows = lower_sd
        highs = upper_sd
    elif type == 'median_quartiles':
        central_measures = medians
        lows = lower_quartiles
        highs = upper_quartiles

    n_groups = len(central_measures)

    if isinstance(line_color, str):
        custom_palette = np.repeat(line_color, n_groups)
    else:
        if len(line_color) != n_groups:
            err1 = "{} groups are being plotted, but ".format(n_groups)
            err2 = "{} colors(s) were supplied in `line_color`.".format(len(line_color))
            raise ValueError(err1 + err2)
        custom_palette = line_color

    try:
        len_offset = len(offset)
    except TypeError:
        offset = np.repeat(offset, n_groups)
        len_offset = len(offset)

    if len_offset != n_groups:
        err1 = "{} groups are being plotted, but ".format(n_groups)
        err2 = "{} offset(s) were supplied in `offset`.".format(len_offset)
        raise ValueError(err1 + err2)

    kwargs['zorder'] = kwargs['zorder']

    for xpos, central_measure in enumerate(central_measures):
        kwargs['color'] = custom_palette[xpos]

        if method == 'sankey_error_bar':
            _xpos = pos[xpos] + offset[xpos]
        else:
            _xpos = xpos + offset[xpos]

        low = lows[xpos]
        low_to_mean = mlines.Line2D([_xpos, _xpos],
                                    [low, central_measure - gap_width],
                                    **kwargs)
        ax.add_line(low_to_mean)

        high = highs[xpos]
        mean_to_high = mlines.Line2D([_xpos, _xpos],
                                     [central_measure + gap_width, high],
                                     **kwargs)
        ax.add_line(mean_to_high)

def check_data_matches_labels(labels, data, side):
    '''
    Function to check that the labels and data match in the sankey diagram. 
    And enforce labels and data to be lists.
    Raises an exception if the labels and data do not match.

    Keywords
    --------
    labels: list of input labels
    data: Pandas Series of input data
    side: string, 'left' or 'right' on the sankey diagram
    '''
    if len(labels > 0):
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise Exception('{0} labels and data do not match.{1}'.format(side, msg))
        
def normalize_dict(nested_dict, target):
    val = {}
    for key in nested_dict.keys():
        val[key] = np.sum([nested_dict[sub_key][key] for sub_key in nested_dict.keys()])
    
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            for subkey in value.keys():
                value[subkey] = value[subkey] * target[subkey]['right']/val[subkey]
    return nested_dict

def single_sankey(left, right, xpos=0, leftWeight=None, rightWeight=None, 
            colorDict=None, leftLabels=None, rightLabels=None, ax=None, 
            width=0.5, alpha=0.65, bar_width=0.2, rightColor=False, align='center'):

    '''
    Make a single Sankey diagram showing proportion flow from left to right
    Original code from: https://github.com/anazalea/pySankey
    Changes are added to normalize each diagram's height to be 1

    Keywords
    --------
    left: NumPy array 
        data on the left of the diagram
    right: NumPy array 
        data on the right of the diagram
        len(left) == len(right)
    xpos: float
        the starting point on the x-axis
    leftWeight: NumPy array
        weights for the left labels, if None, all weights are 1
    rightWeight: NumPy array
         weights for the right labels, if None, all weights are corresponding leftWeight
    colorDict: dictionary of colors for each label
        input format: {'label': 'color'}
    leftLabels: list
        labels for the left side of the diagram. The diagram will be sorted by these labels.
    rightLabels: list
        labels for the right side of the diagram. The diagram will be sorted by these labels.
    ax: matplotlib axes to be drawn on
    aspect: float
        vertical extent of the diagram in units of horizontal extent
    rightColor: bool
        if True, each strip of the diagram will be colored according to the corresponding left labels
    align: bool
        if 'center', the diagram will be centered on each xtick, 
        if 'edge', the diagram will be aligned with the left edge of each xtick
    '''

    # Initiating values
    if ax is None:
        ax = plt.gca()

    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))
    if len(rightWeight) == 0:
        rightWeight = leftWeight

    # Create Dataframe
    if isinstance(left, pd.Series):
        left.reset_index(drop=True, inplace=True)
    if isinstance(right, pd.Series):
        right.reset_index(drop=True, inplace=True)
    dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                              'rightWeight': rightWeight}, index=range(len(left)))
    
    if dataFrame[['left', 'right']].isnull().any(axis=None):
        raise Exception('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.sort(np.r_[dataFrame.left.unique(), dataFrame.right.unique()])[::-1]).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(np.sort(dataFrame.left.unique())[::-1]).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['left'], 'left')

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(np.sort(dataFrame.right.unique())[::-1]).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['right'], 'right')

    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
        fail_color = {0:"grey"}
        colorDict.update(fail_color)
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = "The palette parameter is missing values for the following labels : "
            msg += '{}'.format(', '.join(missing))
            raise ValueError(msg)

    if align not in ("center", "edge"):
        err = '{} assigned for `align` is not valid.'.format(align)
        raise ValueError(err)
    if align == "center":
        try:
            leftpos = xpos - width / 2
        except TypeError as e:
            raise TypeError(f'the dtypes of parameters x ({xpos.dtype}) '
                            f'and width ({width.dtype}) '
                            f'are incompatible') from e
    else: 
        leftpos = xpos

    # Combine left and right arrays to have a pandas.DataFrame in the 'long' format
    left_series = pd.Series(left, name='values').to_frame().assign(groups='left')
    right_series = pd.Series(right, name='values').to_frame().assign(groups='right')
    concatenated_df = pd.concat([left_series, right_series], ignore_index=True)

    # Determine positions of left label patches and total widths
    # We also want the height of the graph to be 1
    leftWidths_norm = defaultdict()
    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD['left'] = (dataFrame[dataFrame.left == leftLabel].leftWeight.sum()/ \
            dataFrame.leftWeight.sum())*(1-(len(leftLabels)-1)*0.02)
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['left']
        else:
            myD['bottom'] = leftWidths_norm[leftLabels[i - 1]]['top'] + 0.02
            myD['top'] = myD['bottom'] + myD['left']
            topEdge = myD['top']
        leftWidths_norm[leftLabel] = myD

    # Determine positions of right label patches and total widths
    rightWidths_norm = defaultdict()
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD['right'] = (dataFrame[dataFrame.right == rightLabel].rightWeight.sum()/ \
            dataFrame.rightWeight.sum())*(1-(len(leftLabels)-1)*0.02)
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['right']
        else:
            myD['bottom'] = rightWidths_norm[rightLabels[i - 1]]['top'] + 0.02
            myD['top'] = myD['bottom'] + myD['right']
            topEdge = myD['top']
        rightWidths_norm[rightLabel] = myD    

    # Total width of the graph
    xMax = width

    # Determine widths of individual strips, all widths are normalized to 1
    ns_l = defaultdict()
    ns_r = defaultdict()
    ns_l_norm = defaultdict()
    ns_r_norm = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
                ].leftWeight.sum()
                
            rightDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
                ].rightWeight.sum()
        factorleft = leftWidths_norm[leftLabel]['left']/sum(leftDict.values())
        leftDict_norm = {k: v*factorleft for k, v in leftDict.items()}
        ns_l_norm[leftLabel] = leftDict_norm
        ns_r[leftLabel] = rightDict
    
    # ns_r should be using a different way of normalization to fit the right side
    # It is normalized using the value with the same key in each sub-dictionary

    ns_r_norm = normalize_dict(ns_r, rightWidths_norm)

    # Plot vertical bars for each label
    for leftLabel in leftLabels:
        ax.fill_between(
            [leftpos + (-(bar_width) * xMax), leftpos],
            2 * [leftWidths_norm[leftLabel]["bottom"]],
            2 * [leftWidths_norm[leftLabel]["bottom"] + leftWidths_norm[leftLabel]["left"]],
            color=colorDict[leftLabel],
            alpha=0.99,
        )
    for rightLabel in rightLabels:
        ax.fill_between(
            [xMax + leftpos, leftpos + ((1 + bar_width) * xMax)], 
            2 * [rightWidths_norm[rightLabel]['bottom']],
            2 * [rightWidths_norm[rightLabel]['bottom'] + rightWidths_norm[rightLabel]['right']],
            color=colorDict[rightLabel],
            alpha=0.99
        )

    # Plot error bars
    error_bar(concatenated_df, x='groups', y='values', ax=ax, offset=0, gap_width_percent=2,
              method="sankey_error_bar",
              pos=[(leftpos + (-(bar_width) * xMax) + leftpos)/2, \
                   (xMax + leftpos + leftpos + ((1 + bar_width) * xMax))/2])
    
    # Plot strips
    for leftLabel, rightLabel in itertools.product(leftLabels, rightLabels):
        labelColor = leftLabel
        if rightColor:
            labelColor = rightLabel
        if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
            # Create array of y values for each strip, half at left value,
            # half at right, convolve
            ys_d = np.array(50 * [leftWidths_norm[leftLabel]['bottom']] + \
                50 * [rightWidths_norm[rightLabel]['bottom']])
            ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
            ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
            ys_u = np.array(50 * [leftWidths_norm[leftLabel]['bottom'] + ns_l_norm[leftLabel][rightLabel]] + \
                50 * [rightWidths_norm[rightLabel]['bottom'] + ns_r_norm[leftLabel][rightLabel]])
            ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
            ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

            # Update bottom edges at each label so next strip starts at the right place
            leftWidths_norm[leftLabel]['bottom'] += ns_l_norm[leftLabel][rightLabel]
            rightWidths_norm[rightLabel]['bottom'] += ns_r_norm[leftLabel][rightLabel]
            ax.fill_between(
                np.linspace(leftpos, leftpos + xMax, len(ys_d)), ys_d, ys_u, alpha=alpha,
                color=colorDict[labelColor], edgecolor='none'
            )
                
def sankeydiag(data, xvar, yvar, left_idx, right_idx, 
                leftLabels=None, rightLabels=None,  
                palette=None, ax=None, 
                one_sankey=False,
                width=0.4, rightColor=False,
                align='center', alpha=0.65, **kwargs):
    '''
    Read in melted pd.DataFrame, and draw multiple sankey diagram on a single axes
    using the value in column yvar according to the value in column xvar
    left_idx in the column xvar is on the left side of each sankey diagram
    right_idx in the column xvar is on the right side of each sankey diagram

    Keywords
    --------
    data: pd.DataFrame
        input data, melted dataframe created by dabest.load()
    xvar, yvar: string.
        x and y columns to be plotted.
    left_idx: str
        the value in column xvar that is on the left side of each sankey diagram
    right_idx: str
        the value in column xvar that is on the right side of each sankey diagram
        if len(left_idx) == 1, it will be broadcasted to the same length as right_idx
        otherwise it should have the same length as right_idx
    leftLabels: list
        labels for the left side of the diagram. The diagram will be sorted by these labels.
    rightLabels: list
        labels for the right side of the diagram. The diagram will be sorted by these labels.
    palette: str or dict
    ax: matplotlib axes to be drawn on
    one_sankey: bool 
        determined by the driver function on plotter.py. 
        if True, draw the sankey diagram across the whole raw data axes
    width: float
        the width of each sankey diagram
    align: str
        the alignment of each sankey diagram, can be 'center' or 'left'
    alpha: float
        the transparency of each strip
    rightColor: bool
        if True, each strip of the diagram will be colored according to the corresponding left labels
    colorDict: dictionary of colors for each label
        input format: {'label': 'color'}
    '''

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    if "width" in kwargs:
        width = kwargs["width"]

    if "align" in kwargs:
        align = kwargs["align"]
    
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    
    if "rightColor" in kwargs:
        rightColor = kwargs["rightColor"]
    
    if "bar_width" in kwargs:
        bar_width = kwargs["bar_width"]

    if ax is None:
        ax = plt.gca()

    allLabels = pd.Series(np.sort(data[yvar].unique())[::-1]).unique()
        
    # Check if all the elements in left_idx and right_idx are in xvar column
    unique_xvar = data[xvar].unique()
    if not all(elem in unique_xvar for elem in left_idx):
        raise ValueError(f"{left_idx} not found in {xvar} column")
    if not all(elem in unique_xvar for elem in right_idx):
        raise ValueError(f"{right_idx} not found in {xvar} column")

    xpos = 0

    # For baseline comparison, broadcast left_idx to the same length as right_idx
    # so that the left of sankey diagram will be the same
    # For sequential comparison, left_idx and right_idx can have anything different 
    # but should have the same length
    if len(left_idx) == 1:
        broadcasted_left = np.broadcast_to(left_idx, len(right_idx))
    elif len(left_idx) != len(right_idx):
        raise ValueError(f"left_idx and right_idx should have the same length")
    else:
        broadcasted_left = left_idx

    if isinstance(palette, dict):
        if not all(key in allLabels for key in palette.keys()):
            raise ValueError(f"keys in palette should be in {yvar} column")
        else: 
            plot_palette = palette
    elif isinstance(palette, str):
        plot_palette = {}
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            plot_palette[label] = colorPalette[i]
    else:
        plot_palette = None

    for left, right in zip(broadcasted_left, right_idx):
        if one_sankey == False:
            single_sankey(data[data[xvar]==left][yvar], data[data[xvar]==right][yvar], 
                            xpos=xpos, ax=ax, colorDict=plot_palette, width=width, 
                            leftLabels=leftLabels, rightLabels=rightLabels, 
                            rightColor=rightColor, bar_width=bar_width,
                            align=align, alpha=alpha)
            xpos += 1
        else:
            xpos = 0 + bar_width/2
            width = 1 - bar_width
            single_sankey(data[data[xvar]==left][yvar], data[data[xvar]==right][yvar], 
                            xpos=xpos, ax=ax, colorDict=plot_palette, width=width, 
                            leftLabels=leftLabels, rightLabels=rightLabels, 
                            rightColor=rightColor, bar_width=bar_width,
                            align='edge', alpha=alpha)

    if one_sankey == False:
        sankey_ticks = [f"{left}\n v.s.\n{right}" for left, right in zip(broadcasted_left, right_idx)]
        ax.get_xaxis().set_ticks(np.arange(len(right_idx)))
        ax.get_xaxis().set_ticklabels(sankey_ticks)
    else:
        sankey_ticks = [broadcasted_left[0], right_idx[0]]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(sankey_ticks)


def horizontal_colormaker(number:int,custom_pal=None,desat_level:float=0.5):
    import seaborn as sns
    import matplotlib.pyplot as plt 

    # If no custom palette is provided, use the default seaborn palette
    if custom_pal is None:
        colors = sns.color_palette(n_colors=number)
    # If a tuple is provided, check it is the right length
    elif isinstance(custom_pal, tuple):
        if len(custom_pal) != number:
            raise ValueError('Number of colors inputted does not equal number of samples')
        else:
            colors = custom_pal
    # If a string is provided, check it is a matplotlib palette
    elif isinstance(custom_pal, str):
        # check it is in the list of matplotlib palettes.
        if custom_pal in plt.colormaps():
            colors = sns.color_palette(custom_pal, number)
        else:
            raise ValueError('The specified `custom_palette` {} is not a matplotlib palette. Please check.'.format(custom_pal))
    else:
        raise TypeError('Incorrect color input format')

    # Desaturate the colors
    desat_colors = [sns.desaturate(c, desat_level) for c in colors] 
    return colors,desat_colors

def horizontal_swarm_plot(data,paired:bool,idx,Num_Exps:int,xvar:str,yvar:str,id_col:str,colors,color_col,minimeta:bool,
                  axes, gap_width_percent:float,raw_marker_size:int,**swarm_kwargs):  
    """The swarm/paired plot function for the horizontal swarm plot

    Args:
        data (df): _description_
        paired (bool): Whether is a paired plot or not
        idx (list): list of the index of the data to be plotted
        Num_Exps (int): The number of experiments
        xvar (str): The independent variable
        yvar (str): The dependent variable
        id_col (str): The id column for paired data
        colors (list of colors): color palette used
        color_col (str): the column name of the color column if provided. Defaults to none.
        minimeta (bool): Whether is a minimeta experiment or not
        axes (ax): axes to plot on
        gap_width_percent (float): The width of the gapped  mean+sd lines. Defaults to 2%.
        raw_marker_size (int): raw data marker size. Defaults to 6.
    """
     
    ## Import Modules
    import pandas as pd
    import numpy as np
    import seaborn as sns

    ##Variables
    swarm_paired_line_alpha = swarm_kwargs['paired_line_alpha']
    swarm_ylabel_show_samplesize = swarm_kwargs['ylabel_show_samplesize']
    swarm_paired_means_offset = swarm_kwargs['paired_means_offset']
    swarm_ylabel_fontsize = swarm_kwargs['ylabel_fontsize']
    dot_alpha = swarm_kwargs['dot_alpha']
    paired_dot = swarm_kwargs['paired_dot']
    paired_dot_size = swarm_kwargs['paired_dot_size']
    paired_dot_alpha = swarm_kwargs['paired_dot_alpha']

    wide_format = True if xvar == None else False
    if color_col == None:
        id_vars = [id_col] if paired == True else None
    else:
        id_vars = [id_col,color_col] if paired == True else [color_col]

    if wide_format == True:
        if paired == False:
            data = pd.melt(data,value_vars=idx[0], id_vars=id_vars)
        else:
            unpacked_idx = [item for sublist in idx for item in sublist]
            data = pd.melt(data,value_vars=unpacked_idx, id_vars=id_vars)
        yvar='value'
        xvar='variable'

    if paired == True:
        Adj_Num_Exps = len(idx) + 1 if minimeta==True else len(idx)
    else:
        Adj_Num_Exps = len(idx[0]) + 1 if minimeta==True else len(idx[0])
    
    ## Unpaired
    if paired == False:
        ordered_labels = idx[0]
        df_list = []
        for i,ypos in zip(ordered_labels,np.arange(0.5,Num_Exps,1)[::-1]):
            _df = data[data[xvar]==i].copy()
            _df['ypos'] = ypos
            df_list.append(_df)
        ordered_df = pd.concat(df_list)
        sns.swarmplot(ax=axes,data=ordered_df, x=yvar,y='ypos',native_scale=True, orient="h",palette=colors[::-1] if color_col == None else colors,
                      alpha=dot_alpha,size=raw_marker_size,hue=color_col)
        axes.set_ylabel('')

    ## Paired
    else:
        data.sort_values(by=[id_col], inplace=True)

        ### Deal with color_col
        if color_col != None:
            color_col_ind = data[color_col].unique()
            color_col_dict = {}
            for n,c in zip(color_col_ind,colors):
                color_col_dict.update({n: c})

        ## Create the data tuples & Mean + SD tuples
        output_x, output_y=[],[]
        means,sd, color_col_names=[],[],[]
        for n,y1,y2 in zip(np.arange(0,Num_Exps,1),np.arange(0.75,Adj_Num_Exps,1)[::-1],np.arange(0.25,Adj_Num_Exps,1)[::-1]):
            output_x.append(np.array([data[data[xvar].str.contains(idx[n][0])][yvar],
                                    data[data[xvar].str.contains(idx[n][1])][yvar]]))
            
            output_y.append(np.array([len(data[data[xvar].str.contains(idx[n][0])])*[y1],
                                    len(data[data[xvar].str.contains(idx[n][1])])*[y2]]))
            
            means.append(np.array([data[data[xvar].str.contains(idx[n][0])][yvar].mean(),
                                data[data[xvar].str.contains(idx[n][1])][yvar].mean()]))
            
            sd.append(np.array([data[data[xvar].str.contains(idx[n][0])][yvar].std(),
                                data[data[xvar].str.contains(idx[n][1])][yvar].std()]))
            if color_col != None:
                color_col_names.append(np.array(data[data[xvar].str.contains(idx[n][0])][color_col]))
            

        ## Plot the pairs of data
        if color_col != None:
            for x, y, cs in zip(output_x,output_y,color_col_names):  
                color_cols = [color_col_dict[i] for i in cs]
                for n,c in zip(range(0,len(x[0])),color_cols):
                    axes.plot([x[0][n],x[1][n]],[y[0][n],y[1][n]],color=c, alpha=swarm_paired_line_alpha)
        else:
            for x, y, c in zip(output_x,output_y,colors):  
                axes.plot(x, y,color=c, alpha=swarm_paired_line_alpha)

        ## Plot dots for each pair of data
        if paired_dot==True:
            for n,y1,y2 in zip(np.arange(0,Num_Exps,1),np.arange(0.75,Adj_Num_Exps,1)[::-1],np.arange(0.25,Adj_Num_Exps,1)[::-1]):
                off = data[data[xvar].str.contains(idx[n][0])][yvar].values
                on = data[data[xvar].str.contains(idx[n][1])][yvar].values
                if color_col == None:
                    axes.plot(off,len(off)*[y1],'o',color=colors[n], markersize = paired_dot_size,alpha=paired_dot_alpha)
                    axes.plot(on,len(on)*[y2], 'o',color=colors[n],markersize = paired_dot_size,alpha=paired_dot_alpha)
                else:
                    color_col_colors = [color_col_dict[i] for i in data[data[xvar].str.contains(idx[n][0])][color_col]]
                    for n,c in zip(range(len(off)),color_col_colors):
                        axes.plot(off[n],y1,'o',color=c, markersize = paired_dot_size,alpha=paired_dot_alpha)
                        axes.plot(on[n],y2, 'o',color=c,markersize = paired_dot_size,alpha=paired_dot_alpha)

                
        ## Plot Mean & SD tuples
        import matplotlib.lines as mlines
        ax_ylims = axes.get_ylim()
        ax_yspan = np.abs(ax_ylims[1] - ax_ylims[0])
        gap_width = ax_yspan * gap_width_percent/100

        if color_col == None:
            for m,n,c in zip(np.arange(0,Num_Exps,1),np.arange(0,Adj_Num_Exps,1)[::-1],colors):
                for a in [0,1]:
                    mean_to_high = mlines.Line2D([means[m][a]+gap_width, means[m][a]+sd[m][a]],[n+swarm_paired_means_offset[a], n+swarm_paired_means_offset[a]],color=c)
                    axes.add_line(mean_to_high) 

                    low_to_mean = mlines.Line2D([means[m][a]-sd[m][a], means[m][a]-gap_width],[n+swarm_paired_means_offset[a], n+swarm_paired_means_offset[a]],color=c)
                    axes.add_line(low_to_mean)
        else:
            for m,n in zip(np.arange(0,Num_Exps,1),np.arange(0,Adj_Num_Exps,1)[::-1]):
                for a in [0,1]:
                    mean_to_high = mlines.Line2D([means[m][a]+gap_width, means[m][a]+sd[m][a]],[n+swarm_paired_means_offset[a], n+swarm_paired_means_offset[a]],color='grey')
                    axes.add_line(mean_to_high) 

                    low_to_mean = mlines.Line2D([means[m][a]-sd[m][a], means[m][a]-gap_width],[n+swarm_paired_means_offset[a], n+swarm_paired_means_offset[a]],color='grey')
                    axes.add_line(low_to_mean)

    ## Parameters for X & Y axes
    axes.set_ylim(0, Adj_Num_Exps)
    axes.set_yticks(np.arange(0.5,Adj_Num_Exps,1))
    axes.tick_params(left=True)
    axes.set_xlabel('Metric')

    yticklabels=[]
    if paired==True:
        for n in np.arange(0,Num_Exps,1):
            if swarm_ylabel_show_samplesize == True:
                ss = len(data[data[xvar].str.contains(idx[n][1])][yvar])
                yticklabels.append(idx[n][1] + ' - ' + '\n' + idx[n][0]  + ' (n='+str(ss)+')')
            else:
                yticklabels.append(idx[n][1] +' - ' + '\n'+ idx[n][0])
        if minimeta==True:
            yticklabels.append('Weighted Mean')                
    else:
        for n in np.arange(0,Num_Exps,1):
            if swarm_ylabel_show_samplesize == True:
                ss = len(data[data[xvar].str.contains(idx[0][n])][yvar])
                yticklabels.append(idx[0][n] + '\n' + ' (n='+str(ss)+')')
            else:
                yticklabels.append(idx[0][n])       
    axes.set_yticklabels(yticklabels[::-1],ma='center',fontsize = swarm_ylabel_fontsize)
    axes.spines[['top', 'right']].set_color(None)


def horizontal_violin_plot(EffectSizeDataFrame,axes,Num_Exps:int,paired:bool,minimeta:bool,colors,color_col,
                           contrast_mean_marker_size:float,halfviolin_alpha:float, contrast_bar, contrast_bar_kwargs, 
                           contrast_dots,contrast_dots_kwargs):
    """Plots the halfviolins for horizontal plot.

    Args:
        EffectSizeDataFrame (dabest effect size dataframe): 
        axes (ax): axes to plot on
        Num_Exps (int): Number of experiments
        paired (bool): Whether the data is paired or not
        minimeta (bool): Whether minimeta analysis or not
        colors (list of colors): color palette used
        color_col (str): the column name of the color column if provided. Defaults to none.
        contrast_mean_marker_size (float): The mean dot marker size. Defaults to 9.
        halfviolin_alpha (float): The alpha value for the halfviolin. Defaults to 0.8.
        contrast_bar (bool): Whether contrast bar is shown or not. Defaults to False.
        contrast_bar_kwargs (dict): Contrast bar kwargs
        contrast_dots (bool): Whether is contrast dots are shown or not. Defaults to False.
        contrast_dots_kwargs (dict): Contrast dots kwargs
    """
    
    ## Import Modules
    import numpy as np
    import dabest
    from dabest import plot_tools
    import matplotlib.patches as mpatches
    import warnings
    from .misc_tools import merge_two_dicts

    ## Variables
    Adj_Num_Exps    = Num_Exps if paired==True else Num_Exps-1
    Adj_MM_Num_Exps = Adj_Num_Exps if minimeta==False else Adj_Num_Exps+1
    paired_colors   = colors if paired==True else colors[1:]
    experiment = EffectSizeDataFrame.effect_size
    idx = EffectSizeDataFrame.idx
    data = EffectSizeDataFrame.dabest_obj.data
    xvar = EffectSizeDataFrame.dabest_obj.x
    yvar = EffectSizeDataFrame.dabest_obj.y
    id_col = EffectSizeDataFrame.dabest_obj.id_col

    ## melt wide data into long format
    wide_format = True if xvar == None else False
    if color_col == None:
        id_vars = [id_col] if paired == True else None
    else:
        id_vars = [id_col,color_col] if paired == True else [color_col]

    if wide_format == True:
        if paired == False:
            data = pd.melt(data,value_vars=idx[0], id_vars=id_vars)
        else:
            unpacked_idx = [item for sublist in idx for item in sublist]
            data = pd.melt(data,value_vars=unpacked_idx, id_vars=id_vars)
        yvar='value'
        xvar='variable'

    ## kwargs
    ### contrast bar
    default_contrast_bar_kwargs = {'color':'grey','alpha':0.1,'zorder':0}
    if contrast_bar_kwargs is None:
        contrast_bar_kwargs = default_contrast_bar_kwargs
    else:
        contrast_bar_kwargs = merge_two_dicts(default_contrast_bar_kwargs,contrast_bar_kwargs)

    ### contrast dots
    default_contrast_dots_kwargs = {'color':None,'alpha':0.5,'size':3}
    if contrast_dots_kwargs is None:
        contrast_dots_kwargs = default_contrast_dots_kwargs
    else:
        contrast_dots_kwargs = merge_two_dicts(default_contrast_dots_kwargs,contrast_dots_kwargs)
    single_color_contrast_dots = contrast_dots_kwargs['color']
    del contrast_dots_kwargs['color']

    ## Select the bootstraps to plot
    bootstraps = [EffectSizeDataFrame.results.bootstraps[n] for n in np.arange(0,Adj_Num_Exps,1)]
    mean_diff  = [EffectSizeDataFrame.results.difference[n] for n in np.arange(0,Adj_Num_Exps,1)]
    bca_low    = [EffectSizeDataFrame.results.bca_low[n] for n in np.arange(0,Adj_Num_Exps,1)]
    bca_high   = [EffectSizeDataFrame.results.bca_high[n] for n in np.arange(0,Adj_Num_Exps,1)]
    ypos       = np.arange(0.25,Adj_MM_Num_Exps,1)[::-1]
    if minimeta==True:
        bootstraps.append(EffectSizeDataFrame.mini_meta_delta.bootstraps_weighted_delta)
        mean_diff.append(EffectSizeDataFrame.mini_meta_delta.difference)
        bca_low.append(EffectSizeDataFrame.mini_meta_delta.bca_low)
        bca_high.append(EffectSizeDataFrame.mini_meta_delta.bca_high)

    ## Plot the halfviolins
    default_violinplot_kwargs = {'widths':1, 'vert':False,'showextrema':False, 'showmedians':False, 'positions': ypos}
    v = axes.violinplot(bootstraps, **default_violinplot_kwargs,)
    dabest.plot_tools.halfviolin(v,  half='top', alpha = halfviolin_alpha)

    ## Plot mean diff and bca_low and bca_high
    axes.plot(mean_diff,ypos, 'k.', markersize = contrast_mean_marker_size)
    axes.plot([bca_low, bca_high], [ypos, ypos],'k', linewidth = 2.5)

    ## Add Grey bar
    if contrast_bar == True:
        for n,y in zip(np.arange(0,Adj_MM_Num_Exps,1),ypos):
            axes.add_patch(mpatches.Rectangle((0,y), mean_diff[n], 0.5, **contrast_bar_kwargs))

    ## Violin colors
    if color_col == None:
        for n,c in zip(np.arange(0,Adj_Num_Exps,1),paired_colors):
            axes.collections[n].set_fc(c)
    else:
        for n in np.arange(0,Adj_Num_Exps,1):
            axes.collections[n].set_fc('grey')

    ## Delta dots?
    if contrast_dots == True:
        if paired == False:
            UserWarning('Contrast dots are not supported for unpaired data. Plotting without...')
        else:
            df_list = []
            for n,ypos in zip(range(len(idx)), np.arange(0,Adj_MM_Num_Exps,1)[::-1]):
                _df = data[data[xvar]==idx[n][0]].copy()
                _df['ypos'] = ypos
                _df['value_exp'] = data[data[xvar]==idx[n][1]][yvar].values
                _df['Diff'] = _df['value_exp'] - _df[yvar]
                df_list.append(_df)
            delta_dot_df = pd.concat(df_list)
            
            if single_color_contrast_dots == None:
                sns.stripplot(ax=axes,data=delta_dot_df, x='Diff',y='ypos',native_scale=True, orient="h",
                            palette=colors[::-1] if color_col == None else colors,hue=color_col,legend=False,**contrast_dots_kwargs)
            else:
                sns.stripplot(ax=axes,data=delta_dot_df, x='Diff',y='ypos',native_scale=True, orient="h",color=single_color_contrast_dots,
                              legend=False,**contrast_dots_kwargs) 
            axes.set_ylabel('')  

    ## Parameters for X & Y axes
    if experiment != 'mean_diff':
        axes.set_xlabel(experiment)
    else:
        axes.set_xlabel('Mean difference')
    axes.set_ylim(0, Num_Exps) if minimeta==False else axes.set_ylim(0, Num_Exps+1)
    axes.set_yticks([])
    axes.tick_params(left=False)
    axes.spines[['top','right','left']].set_color(None)   
    axes.plot([0, 0], [0, Num_Exps], 'k', linewidth = 1) if minimeta==False else axes.plot([0, 0], [0, Num_Exps+1], 'k', linewidth = 1)


def horizontal_table_plot(EffectSizeDataFrame,axes, Num_Exps:int,paired:bool,minimeta:bool,**table_kwargs):
    """Plot the table axes for the horizontal plot

    Args:
        EffectSizeDataFrame : EffectSizeDataFrame
        axes (ax): axes to plot on
        Num_Exps (int): number of experiments
        paired (bool): whether it is a paired analysis
        minimeta (bool): whether it is a minimeta analysis

    kwargs:
        color (str): color of table background
        alpha (float): alpha value of table background
        fontsize (int): font size of table text
        text_color (str): color of table text
    """

    ## Import Modules
    import pandas as pd
    import numpy as np

    ## Variables
    table_color = table_kwargs['color']
    table_alpha = table_kwargs['alpha']
    table_font_size = table_kwargs['fontsize']
    table_text_color = table_kwargs['text_color']
    Adj_Num_Exps = Num_Exps if paired==True else Num_Exps-1
    
    ## Create a table of deltas
    cols=['Δ','N']
    lst = []
    if minimeta==True:
        lst.append([EffectSizeDataFrame.mini_meta_delta.difference,0])
    for n in np.arange(0,Adj_Num_Exps,1)[::-1]:
        lst.append([EffectSizeDataFrame.results.difference[n],0])
    tab = pd.DataFrame(lst, columns=cols)
    
    ## Plot the background color
    axes.axvspan(0, 1, facecolor=table_color, alpha=table_alpha)  

    ## Plot the text
    for i in tab.index:
        axes.text(0.5, i+0.5, "{:+.2f}".format(tab.iloc[i,0]),ha="center", va="center", color=table_text_color,size=table_font_size)
    if paired==False:
        axes.text(0.5, Num_Exps-0.5, "—",ha="center", va="center", color=table_text_color,size=table_font_size)
    ## Parameters for X & Y axes  
    axes.set_yticks([])
    axes.set_ylim(0, Num_Exps) if minimeta==False else axes.set_ylim(0, Num_Exps+1)
    axes.tick_params(left=False, bottom=False)
    axes.spines[['top','bottom','right','left']].set_color(None)
    axes.set_xticks([0.5])
    axes.set_xticklabels([])
    axes.set_xlabel('Δ')