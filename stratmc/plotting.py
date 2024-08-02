import math
import sys

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter as gaussian
from scipy.stats import gaussian_kde

from stratmc.data import accumulation_rate, clean_data
from stratmc.inference import (
    age_range_to_height,
    calculate_lengthscale_stability,
    calculate_proxy_signal_stability,
    count_samples,
    find_gaps,
    map_ages_to_section,
)

pd.options.mode.chained_assignment = None

def proxy_strat(sample_df, ages_df, proxy = 'd13c', plot_constraints = True, plot_excluded_samples = False, cmap = 'Spectral', legend = False, **kwargs):
    """

    Plot stratigraphic data (proxy observations and age constraints) for each section.

    .. plot::

       from stratmc.data import load_object
       from stratmc.plotting import proxy_strat

       example_sample_path = 'examples/example_sample_df'
       example_ages_path = 'examples/example_ages_df'
       sample_df = load_object(example_sample_path)
       ages_df = load_object(example_ages_path)

       proxy_strat(sample_df, ages_df, proxy = 'd13c')

       plt.show()

    Parameters
    ----------

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    proxy: str, optional
        Name of proxy. Defaults to 'd13c'.

    sections: list(str) or numpy.array(str), optional
        List of sections to plot. Defaults to all sections in ``sample_df``.

    plot_constraints: bool, optional
        Whether to plot age constraints as dashed lines. Ages are printed above dashed lines by defalut; to turn off age labels, pass ``print_ages = False``.

    plot_excluded_samples: bool, optional
        Whether to plot proxy observations for excluded samples (``Exclude?`` is ``True`` in ``sample_df``) as red dots. Defaults to ``False``.

    cmap: str, optional
        Name of seaborn color palette to use for sections. Defaults to 'Spectral'.

    legend: bool, optional
        Generate a legend. Defaults to ``False``.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with observations for each section.

    """

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df[~np.isnan(sample_df[proxy])]['section'])

    if 'print_ages' in kwargs:
        print_ages = kwargs['print_ages']
    else:
        print_ages = True

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    cs = {}
    pal = sns.color_palette(cmap,n_colors=len(sections))

    cols = 4
    N = len(sections)
    rows = int(math.ceil(N/cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (10, 4.5*rows))

    for n in range(N):
        section = sections[n]
        cs[section] = pal[n]
        ax = fig.add_subplot(gs[n])

        if len(sample_df[proxy][(sample_df['section'] == section) & (sample_df['Exclude?'] == True)]) > 0:
            if plot_excluded_samples:
                ax.scatter(sample_df[proxy][(sample_df['section'] == section) & (sample_df['Exclude?'] == True)],
                        sample_df['height'][(sample_df['section'] == section) & (sample_df['Exclude?'] == True)],
                        color = 'red',
                        edgecolors = 'k',
                        label = 'Excluded sample')


        ax.scatter(sample_df[proxy][(sample_df['section'] == section) & (sample_df['Exclude?'] != True)],
                   sample_df['height'][(sample_df['section'] == section) & (sample_df['Exclude?'] != True)],
                   color = cs[section],
                   edgecolors = 'k',
                   label = 'Sample')


        xl = ax.get_xlim()
        yl = ax.get_ylim()

        if plot_constraints:
            depositional_age_heights = ages_df['height'][(ages_df['section']==section) & (ages_df['Exclude?'] == False) & (ages_df['intermediate detrital?'] == False)  & (ages_df['intermediate intrusive?'] == False)]
            ax.hlines(depositional_age_heights,
                      xl[0],
                      xl[1],
                      color = cs[section],
                      linestyle = 'dashed',
                      label = 'Depositional age',
                      zorder = 2)

            detrital_age_heights = ages_df['height'][(ages_df['section']==section) & (ages_df['Exclude?'] == False) & (ages_df['intermediate detrital?'] == True)]
            ax.hlines(detrital_age_heights,
                      xl[0],
                      xl[1],
                      color = cs[section],
                      linestyle = 'dotted',
                      label = 'Detrital age',
                      zorder = 2)

            intrusive_age_heights = ages_df['height'][(ages_df['section']==section) & (ages_df['Exclude?'] == False) & (ages_df['intermediate intrusive?'] == True)]
            ax.hlines(intrusive_age_heights,
                      xl[0],
                      xl[1],
                      color = cs[section],
                      linestyle = 'dashdot',
                      label = 'Intrusive age',
                      zorder = 2)

            if legend:
                ax.legend()


        x_range = xl[1] - xl[0]
        y_range = yl[1] - yl[0]

        if print_ages:
            for age, height, sigma, dist_type, param1, param2 in zip(ages_df['age'][(ages_df['section']==section) & (ages_df['Exclude?'] != True)],
                                          ages_df['height'][(ages_df['section']==section) & (ages_df['Exclude?'] != True)],
                                          ages_df['age_std'][(ages_df['section']==section) & (ages_df['Exclude?'] != True)],
                                          ages_df['distribution_type'][(ages_df['section']==section) & (ages_df['Exclude?'] != True)],
                                          ages_df['param_1'][(ages_df['section']==section) & (ages_df['Exclude?'] != True)],
                                          ages_df['param_2'][(ages_df['section']==section) & (ages_df['Exclude?'] != True)]):
                if dist_type == 'Normal':
                    ax.text(xl[0] + x_range * 0.05, # 0.45
                            height + y_range * 0.01,
                            s = f'{age:.1f}' + r'$\pm$' + f'{2*sigma:.1f} Ma',
                            fontsize = 10)

                elif dist_type == 'Uniform':
                    ax.text(xl[0] + x_range * 0.05, # 0.45
                            height + y_range * 0.01,
                            s = f'{param1:.1f} - {param2:.1f} Ma',
                            fontsize = 10)


        ax.set_xlim(xl)

        ax.set_ylabel('Height (m)', fontsize = fs)

        if proxy == 'd13c':
            ax.set_xlabel(r'$\delta^{13}$C$_{\mathrm{carb}} (‰)$', fontsize = fs)
        else:
            ax.set_xlabel(proxy, fontsize = fs)

        ax.set_title(section, fontsize = fs)
        ax.tick_params(bottom = True, top = False, left = True, right = False, direction = 'out', labelsize = fs, width = 1.5)
        ax.locator_params(axis='x', nbins = 5)

    fig.tight_layout()

    plt.subplots_adjust(wspace=0.45)

    return fig

def proxy_inference(sample_df, ages_df, full_trace, legend = True, plot_constraints = False, plot_data = False, plot_excluded_samples = False, plot_mean = False, plot_mle = True, orientation = 'horizontal', marker_size = 20, section_legend = False, section_cmap = 'Spectral', **kwargs):

    """
    Plot the inferred proxy signal over time.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import proxy_inference

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       example_ages_path = 'examples/example_ages_df'
       sample_df = load_object(example_sample_path)
       ages_df = load_object(example_ages_path)

       proxy_inference(sample_df, ages_df, full_trace, proxy = 'd13c', plot_constraints = True, plot_data = True, plot_excluded_samples = False, section_legend = False, plot_mle = True)

       plt.show()


    Parameters
    ----------

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    proxy: str, optional
        Tracer to plot; only required if more than one proxy was included in the inference.

    legend: bool, optional
        Generate a legend. Defaults to ``True``.

    plot_constraints: bool, optional
        Plot age constraints for each section as dashed lines. Defaults to ``False``.

    plot_data: bool, optional
        Plot proxy observations by most likely posterior age. Defaults to ``False``.

    plot_excluded_samples: bool, optional
        Plot proxy observations that were excluded from the inference (``Exclude?`` is ``True`` in ``sample_df``). Defaults to ``False``.

    plot_mean: bool, optional
        Plot the mean as a dashed line. Defaults to ``False``.

    plot_mle: bool, optional
        Plot the maximum likelihood estimate. Defaults to ``True``.

    orientation: str, optional
        Orientation of figure ('horizontal' with age on the x-axis, or 'vertical' with age on the y-axis). Defaults to 'horizontal'.

    marker_size: int, optional
        Size of markers if ``plot_data`` is ``True``. Defaults to 20.

    section_legend: bool, optional
        Include section names in the legend (if ``plot_data`` is ``True``). Defaults to ``False``.

    section_cmap: str, optional
        Name of seaborn color palette to use for sections (if ``plot_data`` is ``True``). Defaults to 'Spectral'.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with the proxy signal inference.

    """

    if orientation == 'horizontal':
        horizontal = True
        vertical = False
    if orientation == 'vertical':
        horizontal = False
        vertical = True

    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    elif horizontal:
        figsize = (9, 5)
    elif vertical:
        figsize = (5, 9)

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']
    else:
        proxy = proxies[0]

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df[~np.isnan(sample_df[proxy])]['section'])

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    ages = full_trace.X_new.X_new.values

    fig=plt.figure(figsize=figsize)

    ax = fig.gca()

    cs = {}
    pal = sns.color_palette(section_cmap,n_colors=len(sections))
    for i in range(len(sections)):
        section = sections[i]
        cs[section] = pal[i]

    min_ages = []
    max_ages = []

    if plot_data: # by max likelihood
        plotted_excluded = False
        plotted_data = False

        for section in sections:
            section_df = sample_df[sample_df['section']==section]
            proxy_idx = (~np.isnan(section_df[proxy])) & (~section_df['Exclude?'].values.astype(bool))
            excluded_idx = (~np.isnan(section_df[proxy])) & (section_df['Exclude?'])

            sec_ages = az.extract(full_trace.posterior)[str(section)+'_ages'].values

            max_like = np.zeros(sec_ages.shape[0])
            for i in np.arange(sec_ages.shape[0]):
                sample_ages = sec_ages[i,:]
                dx = np.linspace(np.min(sample_ages), np.max(sample_ages), 1000)
                max_like[i] = dx[np.argmax(gaussian_kde(sample_ages, bw_method = 1)(dx))]

            if section_legend:
                label = section
            elif not plotted_data:
                label = 'Most likely sample age'
            else:
                label = '_nolegend_'

            if vertical:
                ax.scatter(section_df[proxy][proxy_idx],
                           max_like[proxy_idx],
                           #np.mean(sec_ages, axis = 1),
                           s = marker_size,
                           color = cs[section],
                           label = label,
                           edgecolors = 'k',
                           lw = 0.5,
                           zorder = 3)

                plotted_data = True

                if (plot_excluded_samples) and (len(section_df[proxy][excluded_idx]) > 0):
                    if not plotted_excluded:
                        excluded_label = 'Excluded samples'
                    else:
                        excluded_label = '_nolegend'

                    ax.scatter(section_df[proxy][excluded_idx],
                           max_like[excluded_idx],
                           s = marker_size,
                           color = 'none',
                           label = excluded_label,
                           edgecolors = cs[section],
                           lw = 0.5,
                           zorder = 4)

                    plotted_excluded = True

            if horizontal:
                ax.scatter(max_like[proxy_idx],
                           section_df[proxy][proxy_idx],
                           s = marker_size,
                           color = cs[section],
                           label = label,
                           lw = 0.5,
                           edgecolors = 'k',
                           zorder = 3)

                plotted_data = True

                if (plot_excluded_samples) and (len(section_df[proxy][excluded_idx]) > 0):
                    if not plotted_excluded:
                        excluded_label = 'Excluded samples'
                    else:
                        excluded_label = '_nolegend'

                    ax.scatter(max_like[excluded_idx],
                           section_df[proxy][excluded_idx],
                           s = marker_size,
                           color = 'none',
                           label = excluded_label,
                           edgecolors = cs[section],
                           lw = 0.5,
                           zorder = 4)

                    plotted_excluded = True

        min_ages = np.append(min_ages, np.min(np.mean(sec_ages,axis = 1)))

        max_ages = np.append(max_ages, np.max(np.mean(sec_ages, axis = 1)))

    proxy_pred = az.extract(full_trace.posterior_predictive)['f_pred_' + proxy].values

    lo = np.percentile(proxy_pred, 2.5, axis=1).flatten()
    hi = np.percentile(proxy_pred, 97.5, axis=1).flatten()

    if vertical:
        ax.fill_betweenx(ages.ravel(),
                         hi,
                         lo,
                         color=(.95,.95,.95),
                         label='95% envelope',
                         linestyle = '--',
                         lw = 1.5,
                         edgecolor = 'k',
                         zorder = 0)

    if horizontal:
        ax.fill_between(ages.ravel(),
                        hi,
                        lo,
                        color=(.95,.95,.95),
                        label='95% envelope',
                        linestyle = '--',
                        edgecolor = 'k',
                        lw = 1.5,
                        zorder = 0)

    lo = np.percentile(proxy_pred, 16, axis=1).flatten()
    hi = np.percentile(proxy_pred, 100-16, axis=1).flatten()

    if vertical:
        ax.fill_betweenx(ages.ravel(),
                         hi,
                         lo,
                         color=(.8,.8,.8),
                         label='68% envelope',
                         edgecolor = 'k',
                         lw = 1.5,
                         zorder = 1)
    if horizontal:
        ax.fill_between(ages.ravel(),
                        hi,
                        lo,
                        color=(.8,.8,.8),
                        label='68% envelope',
                        edgecolor = 'k',
                        lw = 1.5,
                        zorder = 1)

    if plot_mean:
        if horizontal:
            ax.plot(ages.ravel(),
                    np.mean(proxy_pred,axis=1).flatten(),
                    color = 'k',
                    linestyle = 'solid',
                    lw = 2,
                    zorder = 2,
                   label = 'Mean ' + str(proxy))

        if vertical:
            ax.plot(np.mean(proxy_pred, axis=1).flatten(),
                    ages.ravel(),
                    color = 'k',
                    linestyle = 'solid',
                    lw = 2,
                    zorder = 2,
                   label = 'Mean ' + str(proxy))

    if plot_mle:
        dy = np.linspace(np.min(proxy_pred), np.max(proxy_pred), 200)
        max_like = np.zeros(ages.size)
        for i in np.arange(ages.size):
            time_slice = proxy_pred[i,:]
            max_like[i] = dy[np.argmax(gaussian_kde(time_slice, bw_method = 1)(dy))]
        max_like = gaussian(max_like, 2)

        if horizontal:
            ax.plot(ages.ravel(),
                    max_like,
                    zorder = 2,
                    color = 'k',
                    linestyle = 'solid',
                    lw = 2,
                    label = 'Most likely ' + str(proxy))

        if vertical:
            ax.plot(max_like,
                    ages.ravel(),
                    zorder = 2,
                    color = 'k',
                    linestyle = 'solid',
                    lw = 2,
                    label = 'Most likely ' + str(proxy))

    xl = ax.get_xlim()
    yl = ax.get_ylim()

    if plot_constraints:
        i = 0
        for section in sections:
            if i == 0:
                label = 'Age constraint'
            else:
                label =  '_nolegend_'
            if vertical:
                ax.hlines(ages_df['age'][ages_df['section']==section],
                          xl[0],
                          xl[1],
                          color = cs[section],
                          linestyle = 'dashed',
                          label = label,
                          zorder = -1)
                ax.set_xlim(xl)

            if horizontal:
                ax.vlines(ages_df['age'][ages_df['section']==section],
                          yl[0],
                          yl[1],
                          color = cs[section],
                          linestyle = 'dashed',
                          label = label,
                          zorder = -1)
                ax.set_ylim(yl)
            i += 1

    min_ages = np.append(min_ages, np.min(ages))
    max_ages = np.append(max_ages, np.max(ages))

    if vertical:
        ax.set_ylabel('Age (Ma)', fontsize = fs)
        if proxy == 'd13c':
            ax.set_xlabel(r'$\delta^{13}$C$_{\mathrm{carb}}$ (‰)', fontsize = fs)
        else:
            ax.set_xlabel(proxy, fontsize = fs)

        ax.set_ylim([np.min(min_ages) - 1, np.max(max_ages) + 1])
        ax.invert_yaxis()

    if horizontal:
        ax.set_xlabel('Age (Ma)', fontsize = fs)
        if proxy == 'd13c':
            ax.set_ylabel(r'$\delta^{13}$C$_{\mathrm{carb}}$ (‰)', fontsize = fs)
        else:
            ax.set_ylabel(proxy, fontsize = fs)

        ax.set_xlim([np.min(min_ages) - 1, np.max(max_ages) + 1])
        ax.invert_xaxis()

    ax.tick_params(bottom = True, top = True, left = True, right = True, direction = 'in', labelsize = fs, width = 1.5)

#     for axis in ['top','bottom','left','right']:
#         ax.spines[axis].set_linewidth(1.5)

    if legend:
        ax.legend(loc='best')

    return fig

def interpolated_proxy_inference(interpolated_df, interpolated_proxy_df, proxy, legend = True, plot_data = False, plot_mle = True, orientation = 'horizontal', section_legend = False, marker_size = 20, section_cmap = 'Spectral', **kwargs):
    """

    Plot interpolated proxy signal over time (by extending the posterior section age models to a new proxy not included in the inference model) using interpolated age models from :py:meth:`extend_age_model() <stratmc.inference.extend_age_model>` and interpolated proxy values from :py:meth:`interpolate_proxy() <stratmc.inference.interpolate_proxy>` in :py:mod:`stratmc.inference`.

    .. plot::

        from stratmc.data import load_object, load_trace
        from stratmc.inference import extend_age_model, interpolate_proxy
        from stratmc.plotting import interpolated_proxy_inference

        full_trace = load_trace('examples/example_docs_trace')
        example_sample_path = 'examples/example_sample_df'
        example_sample_path_d18o = 'examples/example_sample_df_d18o'
        example_ages_path = 'examples/example_ages_df'
        sample_df = load_object(example_sample_path)
        sample_df_d18o = load_object(example_sample_path_d18o)
        ages_df = load_object(example_ages_path)

        interpolated_df = extend_age_model(full_trace, sample_df, ages_df, ['d18o'], new_proxy_df = sample_df_d18o)
        ages_new = full_trace.X_new.X_new.values.ravel()
        interpolated_proxy_df = interpolate_proxy(interpolated_df, 'd18o', ages_new)

        interpolated_proxy_inference(interpolated_df, interpolated_proxy_df, 'd18o')

        plt.show()


    Parameters
    ----------

    interpolated_df: pandas.DataFrame
        :class:`pandas.DataFrame` with interpolated age draws and sample age summary statistics from :py:meth:`extend_age_model() <stratmc.inference.extend_age_model>` in :py:mod:`stratmc.inference`.

    interpolated_proxy_df: pandas.DataFrame
        :class:`pandas.DataFrame` with interpolated proxy values and summary statistics at target ages from :py:meth:`interpolate_proxy() <stratmc.inference.interpolate_proxy>` in :py:mod:`stratmc.inference`.

    proxy: str
        Name of new proxy (must match column name in ``interpolated_proxy_df``).

    legend: bool, optional
        Generate a legend. Defaults to ``True``.

    plot_data: bool, optional
        Plot proxy observations by most likely posterior age. Defaults to ``False``.

    plot_mle: bool, optional
        Plot the maximum likelihood estimate. Defaults to ``True``.

    orientation: str, optional
        Orientation of figure ('horizontal' or 'vertical'). Defaults to 'horizontal'.

    marker_size: int, optional
        Size of markers if ``plot_data`` is ``True``. Defaults to 20.

    section_legend: bool, optional
        Include section names in the legend (if ``plot_data`` is ``True``). Defaults to ``False``.

    section_cmap: str, optional
        Name of seaborn color palette to use for sections (if ``plot_data`` is ``True``). Defaults to 'Spectral'.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with interpolated proxy signal over time.

    """

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(interpolated_df['section'])

    if orientation == 'horizontal':
        horizontal = True
        vertical = False
    if orientation == 'vertical':
        horizontal = False
        vertical = True

    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    elif horizontal:
        figsize = (9, 5)
    elif vertical:
        figsize = (5, 9)

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    fig=plt.figure(figsize=figsize)

    ax = fig.gca()

    cs = {}
    pal = sns.color_palette(section_cmap,n_colors=len(sections))
    for i in range(len(sections)):
        section = sections[i]
        cs[section] = pal[i]

    min_ages = []
    max_ages = []

    if plot_data: # by max likelihood
        for section in sections:
            section_df = interpolated_df[interpolated_df['section']==section]

            if section_legend:
                label = section
            else:
                label = '_nolegend_'

            if vertical:
                ax.scatter(section_df[proxy],
                           section_df['mle'],
                           #np.mean(sec_ages, axis = 1),
                           s = marker_size,
                           color = cs[section],
                           label = label,
                           edgecolors = 'k',
                           lw = 0.5,
                           zorder = 3)

            if horizontal:
                ax.scatter(section_df['mle'],
                           section_df[proxy],
                           s = marker_size,
                           color = cs[section],
                           label = label,
                           lw = 0.5,
                           edgecolors = 'k',
                           zorder = 3)

        sec_ages = section_df['mle'].values

        min_ages = np.append(min_ages, np.min(sec_ages))

        max_ages = np.append(max_ages, np.max(sec_ages))


    if vertical:
        ax.fill_betweenx(interpolated_proxy_df['age'],
                         interpolated_proxy_df['97.5'],
                         interpolated_proxy_df['2.5'],
                         color=(.95,.95,.95),
                         label='95% envelope',
                         linestyle = '--',
                         lw = 1.5,
                         edgecolor = 'k',
                         zorder = 0)

    if horizontal:
        ax.fill_between(interpolated_proxy_df['age'],
                        interpolated_proxy_df['97.5'],
                        interpolated_proxy_df['2.5'],
                        color=(.95,.95,.95),
                        label='95% envelope',
                        linestyle = '--',
                        edgecolor = 'k',
                        lw = 1.5,
                        zorder = 0)


    if vertical:
        ax.fill_betweenx(interpolated_proxy_df['age'],
                         interpolated_proxy_df['84'],
                         interpolated_proxy_df['16'],
                         color=(.8,.8,.8),
                         label='68% envelope',
                         edgecolor = 'k',
                         lw = 1.5,
                         zorder = 1)
    if horizontal:
        ax.fill_between(interpolated_proxy_df['age'],
                        interpolated_proxy_df['84'],
                        interpolated_proxy_df['16'],
                        color=(.8,.8,.8),
                        label='68% envelope',
                        edgecolor = 'k',
                        lw = 1.5,
                        zorder = 1)

    if plot_mle:
        if horizontal:
            ax.plot(interpolated_proxy_df['age'], interpolated_proxy_df['mle'], zorder = 2, color = 'k', linestyle = 'solid', lw = 2, label = 'Most likely ' + str(proxy))
        if vertical:
            ax.plot(interpolated_proxy_df['mle'], interpolated_proxy_df['age'], zorder = 2, color = 'k', linestyle = 'solid', lw = 2, label = 'Most likely ' + str(proxy))

    min_ages = np.append(min_ages, np.min(interpolated_proxy_df['age']))
    max_ages = np.append(max_ages, np.max(interpolated_proxy_df['age']))

    if vertical:
        ax.set_ylabel('Age (Ma)', fontsize = fs)
        ax.set_xlabel(proxy, fontsize = fs)

        ax.set_ylim([np.min(min_ages) - 1, np.max(max_ages) + 1])
        ax.invert_yaxis()

    if horizontal:
        ax.set_xlabel('Age (Ma)', fontsize = fs)
        ax.set_ylabel(proxy, fontsize = fs)

        ax.set_xlim([np.min(min_ages) - 1, np.max(max_ages) + 1])
        ax.invert_xaxis()

    ax.tick_params(bottom = True, top = True, left = True, right = True, direction = 'in', labelsize = fs, width = 1.5)

    if legend:
        ax.legend(loc='best')

    return fig


def age_height_model(sample_df, ages_df, full_trace, include_excluded_samples = True, cmap = 'Spectral', legend = False, **kwargs):
    """
    Generate a posterior age-height plot for each section.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import age_height_model

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       example_ages_path = 'examples/example_ages_df'
       sample_df = load_object(example_sample_path)
       ages_df = load_object(example_ages_path)

       age_height_model(sample_df, ages_df, full_trace)

       plt.show()

    Parameters
    ----------

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sections: list(str) or numpy.array(str), optional
        List of sections to plot. Defaults to all sections in ``sample_df``.

    cmap: str, optional
        Name of seaborn color palette to use for sections. Defaults to 'Spectral'.

    legend: bool, optional
        Generate a legend. Defaults to ``False``.

    include_excluded_samples: bool, optional
        Whether to consider excluded samples (``Exclude?`` is ``True`` in ``sample_df``) whose ages were passively tracked in the inference model. Defaults to ``True``.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with age-height models for each section.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']

    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    cs = {}
    pal = sns.color_palette(cmap, n_colors=len(sections))

    cols = 4
    N = len(sections)
    rows = int(math.ceil(N/cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (11, 2.5*rows))

    for n in range(N):
        section = sections[n]
        cs[section] = pal[n]
        ax = fig.add_subplot(gs[n])

        sec_ages = az.extract(full_trace.posterior)[str(section)+'_ages'].values # posterior_predictive

        section_df = sample_df[sample_df['section']==section]

        lo = np.percentile(sec_ages, 2.5, axis=1).flatten()
        hi = np.percentile(sec_ages, 97.5, axis=1).flatten()

        if include_excluded_samples:
            ax.fill_betweenx(section_df['height'], hi, lo,color=(.95,.95,.95),label=r'2-$\sigma$')

        else:
            included_idx = ~section_df['Exclude?'].values.astype(bool)
            ax.fill_betweenx(section_df['height'][included_idx],
                             hi[included_idx],
                             lo[included_idx],
                             color=(.95,.95,.95),
                             label=r'2-$\sigma$')

        lo = np.percentile(sec_ages,16,axis=1).flatten()
        hi = np.percentile(sec_ages,100-16,axis=1).flatten()

        if include_excluded_samples:
            ax.fill_betweenx(section_df['height'],
                             hi,
                             lo,
                             color=(.8,.8,.8),
                             label=r'1-$\sigma$')
        else:
            ax.fill_betweenx(section_df['height'][included_idx],
                             hi[included_idx],
                             lo[included_idx],
                             color=(.8,.8,.8),
                             label=r'1-$\sigma$')

        if include_excluded_samples:
            ax.plot(np.mean(sec_ages, axis = 1),
                    section_df['height'],
                    color = cs[section],
                    linewidth = 2,
                    linestyle = 'dashed',
                    label = 'Mean')
        else:
            ax.plot(np.mean(sec_ages, axis = 1)[included_idx],
                    section_df['height'][included_idx],
                    color = cs[section],
                    linewidth = 2,
                    linestyle = 'dashed',
                    label = 'Mean')


        ax.errorbar(ages_df['age'][ages_df['section']==section],
                     ages_df['height'][ages_df['section']==section],
                     xerr = 2*ages_df['age_std'][ages_df['section']==section],
                     color = cs[section], fmt = 'none', capsize = 3)

        section_ages_df = ages_df[(ages_df['section']==section) & (~ages_df['Exclude?'])]
        ax.scatter(section_ages_df['age'][(~section_ages_df['intermediate intrusive?']) & (~section_ages_df['intermediate detrital?'])],
                    section_ages_df['height'][(~section_ages_df['intermediate intrusive?']) & (~section_ages_df['intermediate detrital?'])],
                    color = cs[section],
                    label = r'Age ($\pm2\sigma$)')


        if section_ages_df['age'][section_ages_df['intermediate intrusive?']].shape[0] > 0:
            ax.scatter(section_ages_df['age'][section_ages_df['intermediate intrusive?']],
                        section_ages_df['height'][section_ages_df['intermediate intrusive?']],
                       marker = 's',
                        color = cs[section],
                        label = r'Intrusive Age ($\pm2\sigma$)')

        if section_ages_df['age'][section_ages_df['intermediate detrital?']].shape[0] > 0:
            ax.scatter(section_ages_df['age'][section_ages_df['intermediate detrital?']],
                        section_ages_df['height'][section_ages_df['intermediate detrital?']],
                       marker = '^',
                        color = cs[section],
                        label = r'Detrital Age ($\pm2\sigma$)')

        if legend:
            plt.legend(loc = 'upper right',
                             bbox_to_anchor=(1, 1),
                             markerfirst = False,
                             frameon = False,
                             fontsize = 8) #0.95,  0.985

        ax.axis('tight')
        ax.set_xlabel('Age (Ma)', fontsize = fs)
        ax.set_ylabel('Height (m)', fontsize = fs)
        ax.tick_params(bottom = True, top = True, left = True, right = True, direction = 'in', labelsize = fs)
        ax.set_title(section, fontsize = fs)

    fig.tight_layout()

    plt.subplots_adjust(wspace=0.35) ## , hspace=0.5

    return fig

def section_proxy_signal(full_trace, sample_df, ages_df, include_radiometric_ages = False, plot_constraints = False, yax = 'height', legend = False, cmap = 'Spectral', **kwargs):
    """
    Map the posterior proxy signal back to height in each section (using its most likely posterior age model), and plot alongside the proxy observations (plotted by most likely posterior age).

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import section_proxy_signal

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       example_ages_path = 'examples/example_ages_df'
       sample_df = load_object(example_sample_path)
       ages_df = load_object(example_ages_path)

       section_proxy_signal(full_trace, sample_df, ages_df)

       plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    include_radiometric_ages: bool, optional
        Whether to consider radiometric ages in the posterior age model for each section. Defaults to ``False``.

    plot_constraints: bool, optional
        Plot age constraints for each section as dashed lines. Defaults to ``False``.

    yax: str, optional
        Scale for the y-axis ('height' or 'age'). Defaults to 'height'.

    legend: bool, optional
        Generate a legend. Defaults to ``True``.

    cmap: str, optional
        Name of seaborn color palette to use for sections. Defaults to 'Spectral'.

    proxy: str, optional
        Tracer to plot; only required if more than one proxy was included in the inference.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with inferred proxy signal mapped to height in each section.

    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']

    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']

    else:
        proxy = proxies[0]

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        # only want to map signal back to sections that actually have data for this proxy
        sections = np.unique(sample_df.dropna(subset = [proxy], how = 'all')['section'])

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    cs = {}
    pal = sns.color_palette(cmap, n_colors=len(sections))

    N = len(sections)
    cols = np.min([N, 4])
    rows = int(math.ceil(N/cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (2.5 * cols, 4*rows))

    proxy_pred = az.extract(full_trace.posterior_predictive)['f_pred_' + proxy].values
    ages_new = full_trace.X_new.X_new.values

    # calculate MLE
    dy = np.linspace(np.min(proxy_pred), np.max(proxy_pred), 200)
    max_like = np.zeros(ages_new.size)
    for i in np.arange(ages_new.size):
        time_slice = proxy_pred[i,:]
        max_like[i] = dy[np.argmax(gaussian_kde(time_slice, bw_method = 1)(dy))]
    max_like = gaussian(max_like, 2)

    mapped_age_models = map_ages_to_section(full_trace, sample_df, ages_df, sections = sections, include_radiometric_ages = include_radiometric_ages)

    for n in range(N):
        section = sections[n]
        cs[section] = pal[n]
        ax = fig.add_subplot(gs[n])

        ax.tick_params(bottom = True, top = False, left = True, right = False, direction = 'out', labelsize = fs, width = 1.5)

        ax.set_title(section, fontsize = fs)

        # calculate maximum likelihood age for each sample
        section_df = sample_df[sample_df['section']==section]

        proxy_idx = (~np.isnan(section_df[proxy])) & (~section_df['Exclude?'].values.astype(bool))
        excluded_idx = (~np.isnan(section_df[proxy])) & (section_df['Exclude?'])

        section_age_model = mapped_age_models[mapped_age_models['section']==section]
        section_age_vec = section_age_model['age'].values
        above = (ages_new <= np.max(section_age_vec))
        below = (ages_new >= np.min(section_age_vec))

        age_idx = above & below

        # plot data
        if yax == 'height':

            # samples included in inference
            ax.scatter(section_df[proxy].values[proxy_idx],
                        section_df['height'].values[proxy_idx],
                        color = cs[section],
                        edgecolor = 'k',
                       zorder = 3)

            # excluded samples
            ax.scatter(section_df[proxy].values[excluded_idx],
                        section_df['height'].values[excluded_idx],
                        color = 'none',
                        edgecolor = cs[section],
                       zorder = 4)

            # plot posterior proxy signal
            hi = np.percentile(proxy_pred, 97.5, axis=1).flatten()
            lo = np.percentile(proxy_pred, 2.5, axis=1).flatten()

            ax.fill_betweenx(section_age_model['interpolated height'],
                             hi[age_idx],
                             lo[age_idx],
                             color=(.95,.95,.95),
                             label='95% envelope',
                             linestyle = '--',
                             lw = 1.5,
                             edgecolor = 'k',
                             zorder = 0)

            hi = np.percentile(proxy_pred, 100-16, axis=1).flatten()
            lo = np.percentile(proxy_pred, 16, axis=1).flatten()

            ax.fill_betweenx(section_age_model['interpolated height'],
                             hi[age_idx],
                             lo[age_idx],
                             color=(.8,.8,.8),
                             label='68% envelope',
                             linestyle = 'solid',
                             lw = 1.5,
                             edgecolor = 'k',
                             zorder = 1)

            ax.plot(max_like[age_idx],
                    section_age_model['interpolated height'],
                    color = 'k',
                    lw = 2,
                    label = 'Most likely',
                    zorder = 2)

            if plot_constraints:
                section_ages_df = ages_df[(ages_df['section']==section) & (~ages_df['Exclude?'])]
                for h in section_ages_df['height']:
                    ax.axhline(h, color = cs[section], linestyle = 'dashed', zorder = -1)

            if proxy == 'd13c':
                ax.set_xlabel(r'$\delta^{13}$C$_{\mathrm{carb}}$ (‰)', fontsize = fs)

            else:
                ax.set_xlabel(proxy, fontsize = fs)


        elif yax == 'age':

            # plot posterior proxy signal
            hi = np.percentile(proxy_pred, 97.5, axis=1).flatten()
            lo = np.percentile(proxy_pred, 2.5, axis=1).flatten()

            ax.fill_betweenx(ages_new[age_idx],
                             hi[age_idx],
                             lo[age_idx],
                             color=(.95,.95,.95),
                             label='95% envelope',
                             linestyle = '--',
                             lw = 1.5,
                             edgecolor = 'k',
                             zorder = 0)

            hi = np.percentile(proxy_pred, 100-16, axis=1).flatten()
            lo = np.percentile(proxy_pred, 16, axis=1).flatten()

            ax.fill_betweenx(ages_new[age_idx],
                             hi[age_idx],
                             lo[age_idx],
                             color=(.8,.8,.8),
                             label='68% envelope',
                             linestyle = 'solid',
                             lw = 1.5,
                             edgecolor = 'k',
                             zorder = 1)

            ax.plot(max_like[age_idx],
                    ages_new[age_idx],
                    color = 'k',
                    lw = 2,
                    label = 'Most likely',
                    zorder = 2)

            sec_ages = az.extract(full_trace.posterior)[str(section)+'_ages'].values

            max_like_samples = np.zeros(sec_ages.shape[0])
            for i in np.arange(sec_ages.shape[0]):
                sample_ages = sec_ages[i,:]
                dx = np.linspace(np.min(sample_ages), np.max(sample_ages), 1000)
                max_like_samples[i] = dx[np.argmax(gaussian_kde(sample_ages, bw_method = 1)(dx))]

            ax.scatter(section_df[proxy].values[proxy_idx],
                        max_like_samples[proxy_idx],
                        color = cs[section],
                        edgecolor = 'k',
                       zorder = 3)

            ax.scatter(section_df[proxy].values[excluded_idx],
                        max_like_samples[excluded_idx],
                        color = 'none',
                        edgecolor = cs[section],
                       zorder = 4)


            if plot_constraints:
                section_ages_df = ages_df[(ages_df['section']==section)  & (~ages_df['Exclude?'])]
                for age in section_ages_df['age']:
                    ax.axhline(age, color = cs[section], linestyle = 'dashed', zorder = -1)

            ax.invert_yaxis()

            if proxy == 'd13c':
                ax.set_xlabel(r'$\delta^{13}$C$_{\mathrm{carb}}$ (‰)', fontsize = fs)

            else:
                ax.set_xlabel(proxy, fontsize = fs)


        if (n == 0) | ((n % 4) == 0):
            if yax == 'height':
                ax.set_ylabel('Height (m)', fontsize = fs)

            elif yax == 'age':
                ax.set_ylabel('Age (Ma)', fontsize = fs)

        if (n == 0) and legend:
            ax.legend()

    fig.tight_layout()

    return fig


def covariance_hyperparameters(full_trace, figsize = (4, 3.5), **kwargs):

    """
    Plot prior and posterior distributions for the lengthscale (:math:`\\ell`) and variance (:math:`\\sigma`) hyperparameters of the :class:`pymc.gp.cov.ExpQuad <pymc.gp.cov.ExpQuad>` Gaussian process covariance kernel:

    .. math::

        k(x, x') = \\sigma^2 \\mathrm{exp} \\left[-\\frac{(x - x')^2}{2 \\ell^2} \\right]

    .. plot::

       from stratmc.data import load_trace
       from stratmc.plotting import covariance_hyperparameters

       full_trace = load_trace('examples/example_docs_trace')

       covariance_hyperparameters(full_trace)

       plt.show()

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    proxy: str, optional
        Tracer to plot model parameters for (each proxy has a different covariance kernel); only required if more than one proxy was included in the inference.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with prior and posterior model parameters.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']

    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']

    else:
        proxy = proxies[0]

    fig, ax = plt.subplots(2, figsize = figsize)

    sns.kdeplot(full_trace.prior['gp_ls_' + proxy].values.ravel(),
                color='indianred',
                label = 'Prior',
                ax = ax[0],
                cut = 0)

    sns.kdeplot(full_trace.posterior['gp_ls_' + proxy].values.ravel(),
                color = 'indianred',
                label = 'Posterior',
                fill = True,
                edgecolor = 'None',
                ax = ax[0],
                cut = 0)

    ax[0].set_xlabel('Lengthscale', fontsize=fs)
    ax[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax[0].tick_params(labelsize=fs)
    ax[0].set_ylabel("")
    ax[0].set_yticks([])
    ax[0].ticklabel_format(style='plain')
    ax[0].legend()

    sns.kdeplot(full_trace.prior['gp_var_' + proxy].values.ravel(),
                color='#87BED5',
                label = 'prior',
                ax = ax[1],
                cut = 0)

    sns.kdeplot(full_trace.posterior['gp_var_' + proxy].values.ravel(),
                color = '#87BED5',
                label = 'posterior',
                fill = True,
                edgecolor = 'None',
                ax = ax[1],
                cut = 0)

    ax[1].set_xlabel('Variance', fontsize=fs)
    ax[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax[1].tick_params(labelsize=fs)
    ax[1].set_ylabel("")
    ax[1].set_yticks([])
    ax[1].ticklabel_format(style='plain')
    ax[1].legend()

    fig.tight_layout()

    return fig

def section_summary(sample_df, ages_df, full_trace, section, plot_excluded_samples = False, plot_noise_prior = False, plot_offset_prior = False, include_age_constraints_sedrate = True, figsize = (8, 9)):

    """
    For a given section, plot posterior estimates of sample age, sedimentation rate, noise, and offset. Noise and offset terms must be either per-section or global; to plot per-sample noise and offset terms, use :py:meth:`noise_summary() <stratmc.plotting>` and :py:meth:`offset_summary() <stratmc.plotting>`.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import section_summary

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       example_ages_path = 'examples/example_ages_df'
       sample_df = load_object(example_sample_path)
       ages_df = load_object(example_ages_path)

       section_summary(sample_df, ages_df, full_trace, '1')

       plt.show()

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    section: str
        Name of target section.

    plot_excluded_samples: bool, optional
        Plot age estimates for proxy observations that were excluded from the inference (``Exclude?`` is ``True`` in ``sample_df``). Defaults to ``False``.

    plot_noise_prior: bool, optional
        Plot prior distribution for noise term. Defaults to ``False``.

    plot_offset_prior: bool, optional
        Plot prior distribution for offset term. Defaults to ``False``.

    include_age_constraints_sedrate: bool, optional
        Include age constraints in sedimentation rate calculations. Defaults to ``True``.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure summarizing posterior sample ages, sedimentation rate, and posterior noise and offset terms for the input section.
    """

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])


    if str(section) + '_section_noise_' + proxies[0] in list(full_trace.posterior.keys()):
        noise_type = 'section'

    else:
        noise_type = 'groups'

    if str(section) + '_section_offset_' + proxies[0] in list(full_trace.posterior.keys()):
        offset_type = 'section'

    else:
        group_offset_vars = [
        l
        for l in list(full_trace["posterior"].data_vars.keys())
        if f"{'_group_offset_'}" in l
        ]

        if len(group_offset_vars) > 0:
            offset_type = 'groups'
        else:
            offset_type = 'none'

    if (noise_type == 'none') and (offset_type == 'none'):
        n_rows = 3

    elif ((noise_type != 'none') or (offset_type != 'none')) and not ((noise_type != 'none') and (offset_type != 'none')):
        n_rows = 4

    elif ((noise_type != 'none') and (offset_type != 'none')):
        n_rows = 5

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, list(section))

    fig, ax = plt.subplots(n_rows, 1, figsize = figsize, constrained_layout=True, gridspec_kw={'height_ratios': [1, 1, 1.5] + [0.5] * (n_rows - 3)})

    # link axes with age on the x-axis
    ax[0].sharex(ax[1])
    ax[2].sharex(ax[1])

    pal = sns.color_palette("flare", n_colors = np.mean(az.extract(full_trace.posterior)[str(section)+'_ages'].values, axis = 1).size, desat=0.8)
    sns.set_palette(pal)

    samples = np.mean(az.extract(full_trace.posterior)[str(section)+'_ages'], axis = 1).size # posterior_predictive

    sec_ages = az.extract(full_trace.posterior)[str(section)+'_ages'].values # posterior_predictive

    section_df = sample_df[sample_df['section']==section]

    excluded_idx = section_df['Exclude?'].values

    for i in range(samples):
        if not excluded_idx[i]:
            sns.kdeplot(sec_ages[i,:].ravel().astype(float),
                        label = 'posterior',
                        fill = True,
                        edgecolor = None,
                        color = pal[i],
                        alpha = 0.3,
                        ax = ax[0],
                        cut = 0)
        else:
            if plot_excluded_samples:
                sns.kdeplot(sec_ages[i,:].ravel().astype(float),
                            label = 'posterior',
                            fill = True,
                            edgecolor = None,
                            color = pal[i],
                            alpha = 0.3,
                            ax = ax[0],
                            cut = 0)

    yl = ax[0].get_ylim()

    ax[0].vlines(ages_df['age'][ages_df['section']==section],
                 yl[0],
                 yl[1],
                 color = '#8c8684',
                 linestyle = 'dashed',
                 zorder = 3,
                 label = 'Age constraint')

    for i in range(len(ages_df['age'][ages_df['section']==section])):
        ax[0].axvspan(ages_df['age'][ages_df['section']==section].values[i] - ages_df['age_std'][ages_df['section']==section].values[i],
                  ages_df['age'][ages_df['section']==section].values[i] + ages_df['age_std'][ages_df['section']==section].values[i],
                    alpha=0.5,
                    color ='#8c8684')

    ax[0].invert_xaxis()

    ax[0].set_ylim(yl)

    ax[0].set_ylabel('Sample ages', fontsize = 12)
    ax[0].set_yticks([])

    # age constraint posteriors
    constraint_ages = az.extract(full_trace.posterior)[str(section)+'_radiometric_age'].values
    prior_constraint_ages = az.extract(full_trace.prior)[str(section)+'_radiometric_age'].values
    samples = np.mean(constraint_ages, axis = 1).size

    pal = sns.color_palette('crest',n_colors=samples)

    for i in range(samples):
        if i == 0:
            post_label = 'Posterior'
            prior_label = 'Prior'
        else:
            post_label = '_nolegend'
            prior_label = '_nolegend'
        sns.kdeplot(constraint_ages[i,:].ravel(),
                    label = post_label, fill = True, color = pal[i], edgecolor='None', alpha = 0.3, ax = ax[1], cut = 0)
        sns.kdeplot(prior_constraint_ages[i,:].ravel(),
                    label = prior_label, color = pal[i], fill = False, ax = ax[1], cut = 0)

    ax[1].set_ylabel('Age constraints', fontsize = 12)
    ax[1].set_yticks([])

    ax[1].legend()

    # sedimentation rate
    age_bins = 50
    rate_bins = 50

    rate_df = accumulation_rate(full_trace, sample_df, ages_df, sections = section, method = 'successive', include_age_constraints = include_age_constraints_sedrate)

    age_bin_edges = np.linspace(np.min(rate_df['top_age']), np.max(rate_df['base_age']), age_bins)

    rate_bin_edges = np.logspace(np.log10(rate_df['rate'].min()), np.log10(rate_df['rate'].max()), rate_bins)

    # get age bin centers
    age_centers = (age_bin_edges[:-1] + age_bin_edges[1:]) / 2

    age_vec = []
    rate_vec = []

    for rate, base, top in zip(rate_df['rate'], rate_df['base_age'], rate_df['top_age']):
        for i in np.arange(len(age_bin_edges)-1):
            center = age_centers[i]
            current_bin = pd.Interval(age_bin_edges[i], age_bin_edges[i+1], closed = 'left')
            current_sample_pair = pd.Interval(top, base, closed = 'right')
            result = current_bin.overlaps(current_sample_pair)
            if result:
                age_vec.append(center)
                rate_vec.append(rate)

    H, xedges, yedges = np.histogram2d(age_vec, rate_vec, bins=[age_bin_edges, rate_bin_edges], density = False)

    H[H == 0] = np.nan
    H = H/np.nansum(H, axis=1,keepdims=1) # each column (age bin) should sum to 1

    m = ax[2].pcolormesh(xedges, yedges, H.T, cmap = 'jet')

    ax[2].set_yscale('log')
    ax[2].set_ylabel('LOG (Accumulation rate [m/Myr])', fontsize = 12)

    ax[2].set_xlabel('Age (Ma)', fontsize = 12)

    cb = plt.colorbar(m)
    cb.set_label(label='Probability density', fontsize = 12)

    # noise posteriors
    if noise_type != 'none':

        if noise_type == 'section':
            pal = sns.color_palette("deep", n_colors = len(proxies))

        if noise_type == 'groups':
            palettes = ['crest', 'flare', 'Purples', 'Grays', 'Reds', 'Greens', 'YlOrBr', 'Oranges']

        for i in np.arange(len(proxies)):
            proxy = proxies[i]

            if noise_type == 'section':

                posterior_noise_dist = az.extract(full_trace.posterior)[str(section) + '_section_noise_' + proxy].values.ravel()
                prior_noise_dist = az.extract(full_trace.prior)[str(section) + '_section_noise_' + proxy].values.ravel()

                sns.kdeplot(posterior_noise_dist,
                            fill = True,
                            ax = ax[3],
                            edgecolor='none',
                            color = pal[i],
                            alpha = 0.3,
                            label = proxy + ' posterior',
                            cut = 0)

                if plot_noise_prior:
                    sns.kdeplot(prior_noise_dist,
                                fill = False,
                                ax = ax[3],
                                color = pal[i],
                                alpha = 0.3,
                                label = proxy + ' prior',
                                cut = 0)

            elif noise_type == 'groups':

                noise_vars = [
                l
                for l in list(full_trace["posterior"].data_vars.keys())
                if f"{'_group_noise_' + str(proxy)}" in l
                ]

                pal = sns.color_palette(palettes[i], n_colors = len(noise_vars))

                for j, var in enumerate(noise_vars):
                    posterior_noise_dist =  az.extract(full_trace.posterior)[var].values.ravel()

                    sns.kdeplot(posterior_noise_dist,
                                    fill = True,
                                    ax = ax[3],
                                    color = pal[j],
                                    alpha = 0.3,
                                    label = var + ' posterior',
                                    cut = 0
                                    )

                    if plot_noise_prior:
                        prior_noise_dist =  az.extract(full_trace.prior)[var].values.ravel()

                        sns.kdeplot(prior_noise_dist,
                                    fill = False,
                                    ax = ax[3],
                                    color = pal[j],
                                    alpha = 0.3,
                                    label = var + ' prior',
                                    cut = 0
                                    )

                if (len(proxies) > 1) | (noise_type == 'groups'):
                    ax[3].legend()

        ax[3].set_xlabel('Noise', fontsize = 12)
        ax[3].set_yticks([])
        ax[3].set_ylabel(None)

    # offset posteriors
    if offset_type != 'none':

        if offset_type == 'section':
            pal = sns.color_palette("deep", n_colors = len(proxies))

        if offset_type == 'groups':
            palettes = ['crest', 'flare', 'Purples', 'Grays', 'Reds', 'Greens', 'YlOrBr', 'Oranges']

        for i in np.arange(len(proxies)):
            proxy = proxies[i]

            if offset_type == 'section':
                posterior_offset_dist = az.extract(full_trace.posterior)[str(section) + '_section_offset_' + proxy].values.ravel()

                sns.kdeplot(posterior_offset_dist,
                                fill = True,
                                ax = ax[n_rows - 1],
                                edgecolor='none',
                                color = pal[i],
                                alpha = 0.3,
                                label = proxy + ' posterior',
                                cut = 0)

                if plot_offset_prior:
                    prior_offset_dist = az.extract(full_trace.prior)[str(section) + '_section_offset_' + proxy].values.ravel()

                    sns.kdeplot(prior_offset_dist,
                                    fill = False,
                                    ax = ax[n_rows - 1],
                                    color = pal[i],
                                    alpha = 0.3,
                                    label = proxy + ' prior',
                                    cut = 0)

            elif offset_type == 'groups':

                offset_vars = [
                l
                for l in list(full_trace["posterior"].data_vars.keys())
                if f"{'_group_offset_' + str(proxy)}" in l
                ]

                pal = sns.color_palette(palettes[i], n_colors = len(offset_vars))

                for j, var in enumerate(offset_vars):

                    posterior_offset_dist =  az.extract(full_trace.posterior)[var].values.ravel()

                    sns.kdeplot(posterior_offset_dist,
                                fill = True,
                                ax = ax[n_rows - 1],
                                edgecolor='none',
                                color = pal[j],
                                alpha = 0.3,
                                label = var + ' posterior',
                                cut = 0)


                    if plot_offset_prior:
                        prior_offset_dist =  az.extract(full_trace.prior)[var].values.ravel()

                        sns.kdeplot(prior_offset_dist,
                                fill = False,
                                ax = ax[n_rows - 1],
                                color = pal[j],
                                alpha = 0.3,
                                label = var + ' prior',
                                cut = 0)

            if (len(proxies) > 1) | (offset_type == 'groups'):
                    ax[n_rows - 1].legend()

            ax[n_rows - 1].set_xlabel('Offset', fontsize = 12)
            ax[n_rows - 1].set_yticks([])
            ax[n_rows - 1].set_ylabel(None)

    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)

    for axis in ax.ravel():
        axis.tick_params(bottom = True, top = False, left = False, right = False, direction = 'in', labelsize = 12)

    fig.suptitle('Section: ' + str(section), fontsize = 14)

    return fig

def noise_summary(full_trace, **kwargs):

    """
    Plot posterior noise distributions for each section or group of samples (depending on ``noise_type`` used in :py:meth:`build_model() <stratmc.model.build_model>`) for a given proxy. If multiple proxies were included in the inference, pass a ``proxy`` argument to specify which noise terms to plot.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import noise_summary

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       sample_df = load_object(example_sample_path)
       sections = np.unique(sample_df['section'].values)

       noise_summary(full_trace)

       plt.show()

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    proxy: str, optional
        Target proxy; only required if more than one proxy was included in the inference.


    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with noise distributions for each section or group of samples.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']
    else:
        proxy = proxies[0]

    section_noise_vars = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if f"{'_section_noise_' + str(proxy)}" in l
            ]

    if len(section_noise_vars) > 0:
        noise_vars = section_noise_vars

    else:
        noise_vars = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if f"{'_group_noise_' + str(proxy)}" in l
            ]

        if len(noise_vars) == 0:
            sys.exit(f"No noise terms in the model")

    N = len(noise_vars)
    cols = int(np.min([4, N]))

    pal = sns.color_palette("deep", n_colors = len(noise_vars))

    rows = int(math.ceil(N/cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (10, 2.5*rows))

    for n in range(N):

        ax = fig.add_subplot(gs[n])

        var = noise_vars[n]

        ax.set_title(str(var), fontsize = fs)

        posterior_noise_dist = full_trace.posterior[var].values.ravel()

        sns.kdeplot(posterior_noise_dist,
                    ax = ax,
                    fill = True,
                    edgecolor='None',
                    color = pal[n],
                    alpha = 0.5,
                    label = proxy,
                    cut = 0)

        ax.set_ylabel('')
        ax.set_yticks([])
        ax.tick_params(labelsize = fs)

    fig.supxlabel('Noise (%s)' % proxy, fontsize = fs)

    fig.tight_layout()

    plt.subplots_adjust(wspace=0.15, hspace=0.4)

    return fig


def offset_summary(full_trace, **kwargs):

    """

    Plot posterior offset distributions for each section or group of samples (depending on ``offset_type`` used in :py:meth:`build_model() <stratmc.model.build_model>`). If multiple proxies were included in the inference, pass a ``proxy`` argument to specify which offset terms to plot.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import offset_summary

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       sample_df = load_object(example_sample_path)

       offset_summary(full_trace)

       plt.show()

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    proxy: str, optional
        Target proxy; only required if more than one proxy was included in the inference.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with offset distributions for each section or group of samples.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']
    else:
        proxy = proxies[0]

    section_offset_vars = [
        l
        for l in list(full_trace["posterior"].data_vars.keys())
        if f"{'_section_offset_' + str(proxy)}" in l
        ]

    if len(section_offset_vars) > 0:
        offset_vars = section_offset_vars

    else:
        offset_vars = [
        l
        for l in list(full_trace["posterior"].data_vars.keys())
        if f"{'_group_offset_' + str(proxy)}" in l
        ]

        if len(offset_vars) == 0:
            sys.exit(f"No offset terms in the model")


    N = len(offset_vars)
    cols = int(np.min([4, N]))

    pal = sns.color_palette("deep", n_colors = len(offset_vars))

    rows = int(math.ceil(N/cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (10, 2.5*rows))

    for n in range(N):
        ax = fig.add_subplot(gs[n])

        var = offset_vars[n]

        ax.set_title(var, fontsize = fs)

        # shape = samples x draws
        posterior_offset_dist = az.extract(full_trace.posterior)[var].values.ravel()

        sns.kdeplot(posterior_offset_dist,
                        ax = ax,
                        fill = True,
                        edgecolor = 'None',
                        color = pal[n],
                        alpha = 0.3,
                        label = var,
                        cut = 0
                        )

        ax.set_ylabel('')
        ax.set_yticks([])
        ax.tick_params(labelsize = fs)


    fig.supxlabel('Offset (%s)' % proxy, fontsize = fs)

    fig.tight_layout()

    plt.subplots_adjust(wspace=0.15, hspace=0.4)

    return fig

def section_proxy_residuals(full_trace, sample_df, legend = True, cmap = 'Spectral', include_excluded_samples = False, **kwargs):
    """

    Plot the residuals between the observed proxy values for each section and the inferred proxy signal (using the posterior section age models to map the signal back to height in section). Use to check for stratigraphic trends in the residuals, which may give insight to the processes that cause noisy sections to deviate from the inferred common signal. If multiple proxies were included in the inference, pass a ``proxy`` argument.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import section_proxy_residuals

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       sample_df = load_object(example_sample_path)

       section_proxy_residuals(full_trace, sample_df)

       plt.show()

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    legend: bool, optional
        Generate a legend. Defaults to ``True``.

    cmap: str, optional
        Name of seaborn color palette to use for sections. Defaults to 'Spectral'.

    proxy: str, optional
        Target proxy; only required if more than one proxy was included in the inference.

    include_excluded_samples: bool, optional
        Whether to plot the residuals for samples that were excluded from the inference (``Exclude?`` is ``True`` in ``sample_df``). Defaults to ``False``.


    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with residuals for each section.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']

    else:
        proxy = proxies[0]

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df[~np.isnan(sample_df[proxy])]['section'])


    cs = {}
    pal = sns.color_palette(cmap, n_colors=len(sections))

    N = len(sections)
    cols = np.min([N, 4])
    rows = int(math.ceil(N/cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (2.5 * cols, 4*rows))

    proxy_pred = az.extract(full_trace.posterior_predictive)['f_pred_' + proxy].values
    ages_new = full_trace.X_new.X_new.values

    for n in range(N):
        section = sections[n]
        cs[section] = pal[n]
        ax = fig.add_subplot(gs[n])

        ax.set_title(section, fontsize = fs)

        section_sample_df = sample_df[sample_df['section'] == section]

        if include_excluded_samples:
            included_idx = (~np.isnan(section_sample_df[proxy]))
        else:
            included_idx = (~np.isnan(section_sample_df[proxy])) & (~section_sample_df['Exclude?'])

        section_ages = az.extract(full_trace.posterior)[str(section) + '_ages'].values[included_idx, :]

        section_proxy_true = section_sample_df[included_idx][proxy].values
        section_heights = section_sample_df[included_idx]['height'].values

        section_proxy_pred = np.ones((len(section_proxy_true), proxy_pred.shape[1])) * np.nan
        section_residuals =  np.ones((len(section_proxy_true), proxy_pred.shape[1])) * np.nan

        for i in np.arange(proxy_pred.shape[1]):
            section_proxy_pred[:, i] = np.interp(section_ages[:, i], ages_new, proxy_pred[:, i])
            # residual = actual - predicted
            section_residuals[:, i] = section_proxy_true - section_proxy_pred[:, i]

        hi = np.percentile(section_residuals, 97.5, axis = 1).flatten()
        lo = np.percentile(section_residuals, 2.5, axis = 1).flatten()

        ax.fill_betweenx(section_heights,
                             hi,
                             lo,
                             color=(.95,.95,.95),
                             label='95% envelope',
                             linestyle = '--',
                             lw = 1.5,
                             edgecolor = 'k',
                             zorder = 0)

        hi = np.percentile(section_residuals, 100-16, axis=1).flatten()
        lo = np.percentile(section_residuals, 16, axis=1).flatten()

        ax.fill_betweenx(section_heights,
                         hi,
                         lo,
                         color=(.8,.8,.8),
                         label='68% envelope',
                         linestyle = 'solid',
                         lw = 1.5,
                         edgecolor = 'k',
                         zorder = 1)

        # calculate mle
        dy = np.linspace(np.min(section_residuals.ravel()), np.max(section_residuals.ravel()), 200)
        max_like = np.zeros(section_heights.size)
        for i in np.arange(section_heights.size):
            height_slice = section_residuals[i,:]
            max_like[i] = dy[np.argmax(gaussian_kde(height_slice, bw_method = 1)(dy))]

        ax.plot(max_like,
                section_heights,
                color = 'k',
                lw = 2,
                zorder = 2,
                label = 'Most likely')

        ax.set_ylabel('Height (m)', fontsize = fs)

        if legend and n == 0:
            ax.legend(loc = 'upper right')

    fig.supxlabel('Residual ' + str(proxy) + ' value (observed - predicted)', fontsize = fs)

    fig.tight_layout()

    return fig

def sample_ages(full_trace, sample_df, section, plot_excluded_samples = False, cmap = 'viridis'):
    """
    Plot sample age prior and posterior distributions for a given section. Each subplot contains posterior distributions for a different sample.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import sample_ages

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       sample_df = load_object(example_sample_path)
       section = '1'

       sample_ages(full_trace, sample_df, section)

       plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    section: str
        Name of target section.

    plot_excluded_samples: bool, optional
        Plot age distributions for proxy observations that were excluded from the inference (``Exclude?`` is ``True`` in ``sample_df``). Defaults to ``False``.

    cmap: str, optional
        Name of seaborn color palette to use for age distributions. Defaults to 'viridis'.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with prior and posterior sample age distributions.
    """

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    sample_df, _ = clean_data(sample_df, None, proxies, list(section))

    vals = az.extract(full_trace.posterior)[str(section)+'_ages'].values
    prior_vals = az.extract(full_trace.prior)[str(section)+'_ages'].values

    if not plot_excluded_samples:
        included_idx = ~sample_df['Exclude?'].values.astype(bool)
        # shape = (samples x draws)
        vals = vals[included_idx, :]

    else:
        excluded_idx = sample_df['Exclude?'].values

    samples = np.mean(vals, axis = 1).size

    cols = 5
    N = samples
    rows = int(math.ceil(N/cols))

    pal = sns.color_palette(cmap,n_colors=samples)

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (11, 1.5*rows))

    excluded_count = 0
    for i in range(samples):
        ax = fig.add_subplot(gs[i])

        if plot_excluded_samples:
            if excluded_idx[i]:
                linestyle = 'dashed'
                edgecolor = pal[i]
                post_label = 'Posterior (excluded sample)'
                prior_label = 'Prior (excluded sample)'
                excluded_count += 1
            else:
                linestyle = 'solid'
                edgecolor = 'none'
                post_label = 'Posterior'
                prior_label = 'Prior'
        else:
            linestyle = 'solid'
            edgecolor = 'none'
            post_label = 'Posterior'
            prior_label = 'Prior'

        sns.kdeplot(vals[i,:].ravel(),
                    label = post_label, fill = True, color = pal[i], edgecolor=edgecolor, linestyle = linestyle, alpha = 0.3, ax = ax, cut = 0)
        sns.kdeplot(prior_vals[i,:].ravel(),
                label = prior_label, fill = False, color = pal[i], ax = ax, cut = 0)

        ax.tick_params(bottom = True, top = True, left = False, right = False, direction = 'in', labelsize = 12)
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.invert_xaxis()

        if (i == 0) | ((excluded_count == 1) and (excluded_idx[i])):
            ax.legend(frameon = False, loc = 'upper left')

    fig.suptitle('Section: ' + section, fontsize = 14)
    fig.supxlabel('Age (Ma)', fontsize = 12)
    fig.tight_layout()

    return fig

def sample_ages_per_chain(full_trace, sample_df, section, chains = None, plot_prior = False, plot_excluded_samples = False, legend = True, cmap = 'viridis'):
    """
    Plot sample age posterior distributions for a given section, with separate distributions for each chain. Each subplot contains posterior distributions for a different sample. Use to check for posterior multimodality (in this example, each chain has explored a different mode of the posterior age distributions).

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import sample_ages_per_chain

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       sample_df = load_object(example_sample_path)
       section = '1'

       sample_ages_per_chain(full_trace, sample_df, section, chains = [0, 1, 2, 3])

       plt.show()


    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    section: str
        Name of target section.

    chains: list(int) or numpy.array(int); optional
        Indices of chains to include; optional (defaults to all chains).

    plot_prior: bool, optional
        Plot prior distributions for sample ages. Defaults to ``False``.

    plot_excluded_samples: bool, optional
        Plot age distributions for proxy observations that were excluded from the inference (``Exclude?`` is ``True`` in ``sample_df``). Defaults to ``False``.

    legend: bool, optional
        Generate a legend. Defaults to ``True``.

    cmap: str, optional
        Name of seaborn color palette to use for different chains. Defaults to 'viridis'.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with per-chain posterior sample age distributions.

    """

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    sample_df, _ = clean_data(sample_df, None, proxies, list(section))

    # chains x draws x samples
    vals = full_trace.posterior[str(section)+'_ages'].values

    if chains is None:
        chains = np.arange(vals.shape[0])

    prior_vals = az.extract(full_trace.prior)[str(section)+'_ages'].values

    if not plot_excluded_samples:
        included_idx = ~sample_df['Exclude?'].values.astype(bool)
        # shape = (samples x draws)
        vals = vals[:, :, included_idx]

    else:
        excluded_idx = sample_df['Exclude?'].values

    samples = vals.shape[2]

    cols = 5
    N = samples
    rows = int(math.ceil(N/cols))

    cs = {}
    pal = sns.color_palette(cmap, n_colors=len(chains))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (11, 1.5*rows))

    excluded_count = 0

    for i in range(samples):
        ax = fig.add_subplot(gs[i])

        if plot_excluded_samples:
            if excluded_idx[i]:
                linestyle = 'dashed'
                post_label = 'Posterior (excluded sample)'
                prior_label = 'Prior (excluded sample)'
                excluded_count += 1
                lw = 1
            else:
                lw = 0
                linestyle = 'solid'
                post_label = 'Posterior'
                prior_label = 'Prior'

        else:
            lw = 0
            linestyle = 'solid'
            post_label = 'Posterior'
            prior_label = 'Prior'

        for m in range(len(chains)):
            cs[m] = pal[m]

            sns.kdeplot(vals[chains[m], :, i].ravel(),
                        label = post_label + ', chain ' + str(chains[m]), fill = True, color = cs[m], alpha = 0.3, ax = ax, lw = lw, linestyle = linestyle, cut = 0)

        if plot_prior:
            sns.kdeplot(prior_vals[i,:].ravel(),
                        label = prior_label, fill = False, color = 'k', ax = ax, cut = 0)

        ax.tick_params(bottom = True, top = True, left = False, right = False, direction = 'in', labelsize = 12)
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.invert_xaxis()

        if legend:
            if (i == 0) | ((excluded_count == 1) and (excluded_idx[i])):
                ax.legend(frameon = False, loc = 'upper left')

    fig.suptitle('Section: ' + section, fontsize = 14)
    fig.supxlabel('Age (Ma)', fontsize = 12)
    fig.tight_layout()

    return fig

def age_constraints(full_trace, section, cmap = 'viridis', **kwargs):
    """

    For a given section, plot prior and posterior age distributions for all depositional age constraints (and limiting age constraints that provide a minimum or maximum age for the entire section). To plot intermediate limiting ages (i.e., detrital and intrusive age constraints in the middle of a section), use :py:meth:`limiting_age_constraints() <stratmc.plotting.limiting_age_constraints>`.

    .. plot::

       from stratmc.data import load_trace
       from stratmc.plotting import age_constraints

       full_trace = load_trace('examples/example_docs_trace')
       section = '1'

       age_constraints(full_trace, section)

       plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    section: str
        Name of target section.

    cmap: str, optional
        Name of seaborn color palette to use for age distributions. Defaults to 'viridis'.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with prior and posterior depositional age constraint distributions.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    vals = az.extract(full_trace.posterior)[str(section)+'_radiometric_age'].values
    prior_vals = az.extract(full_trace.prior)[str(section)+'_radiometric_age'].values

    # if only 1 age constraint (should only occur if one the min or max age is shared), reshape for plotting
    # (this should actually never happen -- even if shared, the age constraints end up in the section_radiometric_age RV in the model)
    if vals.ndim == 1:
            vals = np.reshape(vals, (1, len(vals)))

    if prior_vals.ndim == 1:
        prior_vals = np.reshape(prior_vals, (1, len(prior_vals)))

    samples = np.mean(vals, axis = 1).size

    cols = np.min([4, samples])
    N = samples
    rows = int(math.ceil(N/cols))

    pal = sns.color_palette(cmap, n_colors=samples)

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (11, 2*rows))

    for i in range(samples):
        ax = fig.add_subplot(gs[i])
        sns.kdeplot(vals[i,:].ravel(),
                    label = 'Posterior', fill = True, color = pal[i], edgecolor='None', alpha = 0.3, ax = ax, cut = 0)
        sns.kdeplot(prior_vals[i,:].ravel(),
                    label = 'Prior', fill = False, color = pal[i], ax = ax, cut = 0)

        ax.set_ylabel('')
        ax.set_yticklabels([])

        ax.invert_xaxis()
        ax.tick_params(bottom = True, top = True, left = False, right = False, direction = 'in', labelsize = fs)

        if i == 0:
            ax.legend(frameon = False, loc = 'upper left')

    fig.suptitle('Section: ' + section, fontsize = fs + 2)
    fig.supxlabel('Age (Ma)', fontsize = fs)

    fig.tight_layout()

    return fig

def limiting_age_constraints(full_trace, sample_df, ages_df, section, cmap = 'viridis', **kwargs):
    """
    For a given section, plot prior and posterior age distributions for all intermediate limiting (i.e., detrital and intrusive ages in the middle of a section) age constraints. To plot depositional age constraints (and limiting age constraints that provide a minimum or maximum age for the entire section), use :py:meth:`age_constraints() <stratmc.plotting.age_constraints>`.

    .. plot::

        from stratmc.data import load_object, load_trace
        from stratmc.plotting import limiting_age_constraints

        full_trace = load_trace('examples/example_docs_limiting_ages_trace')
        example_sample_path = 'examples/example_sample_df_limiting_ages'
        example_ages_path = 'examples/example_ages_df_limiting_ages'
        sample_df = load_object(example_sample_path)
        ages_df = load_object(example_ages_path)
        section = '1'

        limiting_age_constraints(full_trace, sample_df, ages_df, section)

        plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    section: str
        Name of target section.

    cmap: str, optional
        Name of seaborn color palette to use for age distributions. Defaults to 'viridis'.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with prior and posterior limiting age constraint distributions.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, [section])

    detrital_df = ages_df[(ages_df['section']==section) & (ages_df['intermediate detrital?'])]
    intrusive_df =  ages_df[(ages_df['section']==section) & (ages_df['intermediate intrusive?'])]

    detrital_named = []
    if detrital_df.shape[0] > 0:
        # grab names from list of model variables (will only include constriants that aren't shared)
        detrital_vars = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (l[0:len(section)] == section) and (f"{'detrital_age_'}" in l)
            ]

        detrital_named += [False] * len(detrital_vars)

        # if constraints are shared, grab those names too
        detrital_vars += list(detrital_df[detrital_df['shared?']]['name'].values)

        detrital_named += [True] * len(list(detrital_df[detrital_df['shared?']]['name'].values))

    else:
        detrital_vars = []

    intrusive_named = []
    if intrusive_df.shape[0] > 0:

        # grab names from list of model variables (will only include constriants that aren't shared)
        intrusive_vars = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (l[0:len(section)] == section) and (f"{'intrusive_age_'}" in l)
            ]

        intrusive_named += [False] * len(intrusive_vars)

        # if constraints are shared, grab those names too
        intrusive_vars += list(intrusive_df[intrusive_df['shared?']]['name'].values)
        intrusive_named += [True] * len(list(intrusive_df[intrusive_df['shared?']]['name'].values))

    else:
        intrusive_vars = []

    # number of age constraints to plot
    combined_vars = detrital_vars + intrusive_vars
    combined_named = detrital_named + intrusive_named

    if len(combined_vars) == 0:
            sys.exit(f"Section {section} has no intermediate limiting age constraints")

    N = len(combined_vars)

    cols = np.min([4, N])
    rows = int(math.ceil(N/cols))

    pal = sns.color_palette(cmap, n_colors=N)

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (11, 2*rows))

    for i in range(N):
        var = combined_vars[i]

        named = combined_named[i]
        if named:
            title = var
        elif i < len(detrital_vars):
            title = 'Detrital ' + str(i)
        else:
            title = 'Intrusive ' + str(i - len(detrital_vars))

        ax = fig.add_subplot(gs[i])

        ax.set_title(title, fontsize = fs)

        # distributions for detrital and intrsive ages are always 1d (different objects created for each constraint)
        vals = az.extract(full_trace.posterior)[var].values
        prior_vals = az.extract(full_trace.prior)[var].values

        sns.kdeplot(vals.ravel(),
                    label = 'Posterior', fill = True, color = pal[i], edgecolor='None', alpha = 0.3, ax = ax, cut = 0)
        sns.kdeplot(prior_vals.ravel(),
                    label = 'Prior', fill = False, color = pal[i], ax = ax, cut = 0)

        ax.set_ylabel('')
        ax.set_yticklabels([])

        ax.invert_xaxis()
        ax.tick_params(bottom = True, top = True, left = False, right = False, direction = 'in', labelsize = fs)

        if i == 0:
            ax.legend(frameon = False, loc = 'upper left')

    fig.suptitle('Section: ' + section, fontsize = fs + 2)
    fig.supxlabel('Age (Ma)', fontsize = fs)

    fig.tight_layout()

    return fig

def sadler_plot(full_trace, sample_df, ages_df, method = 'density', duration_bins = 50, rate_bins = 50, scale = 'log', include_age_constraints = True, density_cmap = 'jet', section_cmap = 'Spectral', figsize = (6, 4), **kwargs):
    """
    Plot accumulation rate against duration.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import sadler_plot

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       example_ages_path = 'examples/example_ages_df'
       sample_df = load_object(example_sample_path)
       ages_df = load_object(example_ages_path)

       sadler_plot(full_trace, sample_df, ages_df)

       plt.show()


    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    method: str, optional
        Plot as a 2D histogram ('density') or as a scatter plot ('scatter'). The density plot will combine data for all sections in ``sections``, while a scatter plot will assign a unique color to each section. Defaults to 'density'.

    duration_bins: int, optional
        Number of bin edges to use for the duration data. Defaults to 50.

    rate_bins: int, optional
        Number of bin edges to use for the rate data. Defaults to 50.

    scale: str, optional
        Scaling for x- and y-axes ('linear' or 'log'). Defaults to 'log'.

    sections: list(str) numpy.array(str), optional
        List of target sections. Defaults to all sections in ``sample_df``.

    include_age_constraints: bool, optional
        Include age constraints in sedimentation rate calculations. Defaults to ``False``.

    density_cmap: str, optional
        Name of matplotlib colormap to use for probability density if ``method`` is 'density`. Defaults to 'jet'.

    section_cmap: str, optional
        Name of seaborn color palette to use for sections if ``method`` is 'scatter`. Defaults to 'Spectral'.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with sediment accumulation rate plotted against duration.

    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        # any sections that have no relevant proxy observations would not have ages tracked in the trace
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    rate_df = accumulation_rate(full_trace, sample_df, ages_df, method = 'all', sections = sections, include_age_constraints = include_age_constraints)

    dur = rate_df['duration'].values
    rate = rate_df['rate'].values

    fig = plt.figure(figsize = figsize)
    ax = fig.gca()

    if method == 'density':
        if scale == 'log':
            x_bins = np.logspace(np.log10(dur.min()), np.log10(dur.max()), duration_bins)
            y_bins = np.logspace(np.log10(rate.min()), np.log10(rate.max()), rate_bins)

        else:
            x_bins = np.linspace(dur.min(), dur.max(), duration_bins)
            y_bins = np.linspace(rate.min(),rate.max(), rate_bins)

        H, xedges, yedges = np.histogram2d(dur, rate, bins=[x_bins, y_bins], density = False) # don't normalize during this step - calculation not correct if scale = log

        H[H == 0] = np.nan
        H = H/len(dur) # normalize

        m = ax.pcolormesh(xedges, yedges, H.T, cmap = density_cmap)

        cb = plt.colorbar(m)
        cb.set_label(label='Probability density', fontsize = fs)

    elif method == 'scatter':
        pal = sns.color_palette(section_cmap, n_colors = len(sections))
        for i in np.arange(len(sections)):
            section = sections[i]
            ax.scatter(rate_df[rate_df['section']==section]['duration'].values, rate_df[rate_df['section']==section]['rate'].values,
                       s = 5,
                       edgecolor = pal[i],
                       facecolor = 'none',
                      alpha = 0.5,
                      label = section)
        ax.legend()

    if scale == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('LOG (Duration [yr])', fontsize = fs)
        ax.set_ylabel('LOG (Accumulation rate [mm/yr])', fontsize = fs)

    else:
        ax.set_xlabel('Duration (yr)', fontsize = fs)
        ax.set_ylabel('Accumulation rate (mm/yr)', fontsize = fs)

    ax.tick_params(labelsize = fs)

    fig.tight_layout()

    return fig


def accumulation_rate_stratigraphy(full_trace, sample_df, ages_df, age_bins = 50, age_bin_edges = [], rate_bins = 50, rate_bin_edges = [], rate_scale = 'log', include_age_constraints = True, cmap = 'jet', figsize = (8, 4), **kwargs):
    """
    Plot the probability density of sediment accumulation rates (calculated between successive samples) through time. Probability density for each age bin sums to 1. Unless a ``sections`` argument is passed, includes accumulation rates for all sections.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import accumulation_rate_stratigraphy

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       example_ages_path = 'examples/example_ages_df'
       sample_df = load_object(example_sample_path)
       ages_df = load_object(example_ages_path)

       accumulation_rate_stratigraphy(full_trace, sample_df, ages_df, sections = ['1'])

       plt.show()

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    age_bins: int, optional
        Number of bin edges for age data (if ``age_bin_edges`` not provided). Defaults to 50.

    rate_bins: int, optional
        Number of bin edges for rate data (if ``rate_bin_edges`` not provided). Defaults to 50.

    age_bin_edges: list(float) or numpy.array(float), optional
        List or array of bin edges for the age data.

    rate_bin_edges: list(float) numpy.array(float), optional
        List or array of bin edges for the rate data.

    rate_scale: str, optional
        Scaling for rate ('linear' or 'log'). Defaults to 'log'.

    include_age_constraints: bool, optional
        Include age constraints in sedimentation rate calculations. Defaults to ``True``.

    sections: list(str) or numpy.array(str), optional
        Section(s) to include. If not provided, combines data from all sections in ``sample_df``.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with accumulation rate probability density through time.

    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    rate_df = accumulation_rate(full_trace, sample_df, ages_df, sections = sections, method = 'successive', include_age_constraints = include_age_constraints)

    if len(age_bin_edges) == 0:
        age_bin_edges = np.linspace(np.min(rate_df['top_age']), np.max(rate_df['base_age']), age_bins)

    if len(rate_bin_edges) == 0:
        if rate_scale == 'log':
            rate_bin_edges = np.logspace(np.log10(rate_df['rate'].min()), np.log10(rate_df['rate'].max()), rate_bins)
        elif rate_scale == 'linear':
            rate_bin_edges = np.linspace(rate_df['rate'].min(), rate_df['rate'].max(), rate_bins)

    # get age bin centers
    age_centers = (age_bin_edges[:-1] + age_bin_edges[1:]) / 2

    age_vec = []
    rate_vec = []

    for rate, base, top in zip(rate_df['rate'], rate_df['base_age'], rate_df['top_age']):
        for i in np.arange(len(age_bin_edges)-1):
            center = age_centers[i]
            current_bin = pd.Interval(age_bin_edges[i], age_bin_edges[i+1], closed = 'left')
            current_sample_pair = pd.Interval(top, base, closed = 'right')
            result = current_bin.overlaps(current_sample_pair)
            if result:
                age_vec.append(center)
                rate_vec.append(rate)


    fig = plt.figure(figsize = figsize)
    ax = fig.gca()

    H, xedges, yedges = np.histogram2d(age_vec, rate_vec, bins=[age_bin_edges, rate_bin_edges], density = False)

    H[H == 0] = np.nan
    H = H/np.nansum(H, axis=1,keepdims=1) # each column (age bin) should sum to 1

    m = ax.pcolormesh(xedges, yedges, H.T, cmap = cmap)

    if rate_scale == 'log':
        ax.set_yscale('log')
        ax.set_ylabel('LOG (Accumulation rate [m/Myr])', fontsize = fs)

    elif rate_scale == 'linear':
        ax.set_ylabel('Accumulation rate (m/Myr)', fontsize = fs)

    ax.invert_xaxis()
    ax.set_xlabel('Age (Ma)', fontsize = fs)

    cb = plt.colorbar(m)
    cb.set_label(label='Probability density', fontsize = fs)

    return fig


def section_age_range(full_trace, sample_df, ages_df, lower_age, upper_age, legend = True, section_cmap = 'Spectral', **kwargs):
    """
    Plot the stratigraphic interval corresponding to a given age range (based on the posterior section age models). If ``sections`` is not provided, includes each section that overlaps the target age range.

    .. plot::

       from stratmc.data import load_object, load_trace
       from stratmc.plotting import section_age_range

       full_trace = load_trace('examples/example_docs_trace')
       example_sample_path = 'examples/example_sample_df'
       example_ages_path = 'examples/example_ages_df'
       sample_df = load_object(example_sample_path)
       ages_df = load_object(example_ages_path)

       section_age_range(full_trace, sample_df, ages_df, 120, 130)

       plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    lower_age: float
        Lower bound (youngest) of the target age interval.

    upper_age: float
        Upper bound (oldest) of the target age interval.

    legend: bool, optional
        Generate a legend. Defaults to ``True``.

    cmap: str, optional
        Name of seaborn color palette to use for sections. Defaults to 'Spectral'.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Tracer stratigraphy for each section, with the stratigraphic interval corresponding to the input age range highlighted.

    """

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']
    else:
        proxy = proxies[0]

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']

    else:
        fs = 12

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    stats_df = age_range_to_height(full_trace, sample_df, ages_df,lower_age, upper_age, **kwargs)

    sections = stats_df.index.tolist()

    cs = {}
    pal = sns.color_palette(section_cmap, n_colors=len(sections))

    cols = 4
    N = len(sections)
    rows = int(math.ceil(N/cols))

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize = (10, 4.5*rows))

    for n in range(N):
        section = sections[n]
        cs[section] = pal[n]
        ax = fig.add_subplot(gs[n])

        section_df = sample_df[sample_df['section']==section]

        ax.scatter(section_df[proxy], section_df['height'], color = cs[section], edgecolor = 'k', zorder = 3)

        # 0 is the top height, -1 is the bottom height
        ax.axhspan(stats_df.loc[section]['base_2.5'], stats_df.loc[section]['top_97.5'], zorder = 0, color = (0.95, 0.95, 0.95), label='95% envelope')
        ax.axhspan(stats_df.loc[section]['base_16'], stats_df.loc[section]['top_84'], zorder = 1, color = (0.8, 0.8, 0.8), label='68% envelope')
        ax.axhline(stats_df.loc[section]['base_mle'], color = 'darkgray', linestyle = 'solid', zorder = 2)
        ax.axhline(stats_df.loc[section]['top_mle'], color = 'darkgray', linestyle = 'solid', zorder = 2, label = 'Most likely range')

        ax.set_ylabel('Height (m)', fontsize = fs)
        if proxy == 'd13c':
            ax.set_xlabel(r'$\delta^{13}$C$_{\mathrm{carb}} (‰)$', fontsize = fs)
        else:
            ax.set_xlabel(proxy, fontsize = fs)

        ax.set_title(section, fontsize = fs)
        ax.tick_params(labelsize=fs)
        ax.locator_params(axis='x', nbins=5)

        if legend:
            if n == 0:
                ax.legend(loc = 'upper left')

    fig.tight_layout()

    plt.subplots_adjust(wspace=0.4)

    return fig

def proxy_data_gaps(full_trace, time_grid = None, yaxis = 'percentage', figsize = (6, 3.5), **kwargs):

    """
    For a set of discrete time bins, shows the number of draws from the posterior where there are no proxy observations (i.e., where there are temporal `gaps` in the proxy data). This plot can be used to determine where additional observations are needed to improve the inference.

    .. plot::

        from stratmc.data import load_trace
        from stratmc.plotting import proxy_data_gaps

        full_trace = load_trace('examples/example_docs_trace')

        proxy_data_gaps(full_trace, time_grid = full_trace.X_new.X_new.values[::2])

        plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    time_grid: np.array, optional
        Time bin edges; if not provided, defaults to the ``ages`` array passed to :py:meth:`get_trace() <stratmc.inference.get_trace>`.

    yaxis: str, optional
        Set y-axis to percentage of posterior draws without observations ('percentage') or to the number of posterior draws without observations ('count`). Defaults to 'percentage`.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with bar plot of number of posterior draws with gaps for each time bin.

    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    if time_grid is None:
        time_grid = full_trace.X_new.X_new.values

    age_gaps, grid_centers, grid_widths, n =  find_gaps(full_trace, time_grid = time_grid)

    fig = plt.figure(figsize = figsize)

    ax = fig.gca()

    if yaxis == 'percentage':
        ax.bar(grid_centers, height = 100 * age_gaps/n, width = grid_widths, color = 'lightgray', edgecolor = 'k', lw = 0.5)
        ax.set_ylabel('% solutions without observations', fontsize = fs)

    elif yaxis == 'count':
        ax.bar(grid_centers, height = age_gaps, width = grid_widths, color = 'lightgray', edgecolor = 'k', lw = 0.5)
        ax.set_ylabel('# solutions without observations', fontsize = fs)

    ax.set_xlabel('Age (Ma)', fontsize = fs)
    ax.tick_params(labelsize = fs)

    ax.invert_xaxis()

    fig.tight_layout()

    return fig

def proxy_data_density(full_trace, time_grid = None, figsize = (6, 3.5), **kwargs):
    """
    Plot the mean number proxy observations in discrete time bins (averaged over all posterior draws). This plot can be used to determine where proxy observations are relatively sparse, and additional observations may improve the inference.

    .. plot::

        from stratmc.data import load_trace
        from stratmc.plotting import proxy_data_density

        full_trace = load_trace('examples/example_docs_trace_data')

        proxy_data_density(full_trace, time_grid = full_trace.X_new.X_new.values[::2])

        plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    time_grid: np.array, optional
        Time bin edges; if not provided, defaults to the ``ages`` array passed to :py:meth:`get_trace() <stratmc.inference.get_trace>`.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with bar plot of mean number of observations in each time bin.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    if time_grid is None:
        time_grid = full_trace.X_new.X_new.values

    sample_counts, grid_centers, grid_widths, n =  count_samples(full_trace, time_grid = time_grid)

    fig = plt.figure(figsize = figsize)

    ax = fig.gca()

    ax.bar(grid_centers, height = sample_counts/n, width = grid_widths, color = 'lightgray', edgecolor = 'k', lw = 0.5)

    ax.set_ylabel('Average # observations', fontsize = fs)
    ax.set_xlabel('Age (Ma)', fontsize = fs)
    ax.tick_params(labelsize = fs)

    ax.invert_xaxis()

    fig.tight_layout()

    return fig

def lengthscale_traceplot(full_trace, chains = None, legend = True, figsize = (5, 3.5), **kwargs):
    """
    Generate trace plot (parameter value vs. step in Markov chain) for the :class:`pymc.gp.cov.ExpQuad <pymc.gp.cov.ExpQuad>` covariance kernel lengthscale. By default, includes all chains; to plot the draws for only a subset of chains, past list of chain indices to ``chains``. Use to check for posterior multimodality.

    .. plot::

        from stratmc.data import load_trace
        from stratmc.plotting import lengthscale_traceplot

        full_trace = load_trace('examples/example_docs_trace')

        lengthscale_traceplot(full_trace, chains = [0, 1, 2, 3])

        plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    chains: list(int) or numpy.array(int); optional
        Indices of chains to plot; optional (defaults to all chains).

    proxy: str, optional
        Target proxy; only required if more than one proxy was included in the inference.

    legend: bool, optional
        Generate a legend. Defaults to ``True``.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure with lengthscale trace plot.
    """


    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']

    else:
        proxy = proxies[0]

    fig = plt.figure(figsize = figsize)
    ax = fig.gca()

    if chains is None:
        post_ls = full_trace.posterior['gp_ls_' +str(proxy)].values
        chains = np.arange(post_ls.shape[0])

    else:
        post_ls = full_trace.sel(chain = chains).posterior['gp_ls_' +str(proxy)].values

    sns.reset_defaults()

    for i in np.arange(post_ls.shape[0]):
        ax.plot(np.arange(post_ls.shape[1]),
                post_ls[i, :, :],
                lw = 0.5,
                label = 'Chain ' + str(chains[i]))

    ax.set_xlabel('Draw', fontsize = fs)
    ax.set_ylabel('Lengthscale (%s)' % proxy, fontsize = fs)

    if legend:
        ax.legend()

    ax.tick_params(labelsize = fs)

    fig.tight_layout()

    return fig

def lengthscale_stability(full_trace, figsize = (5, 3.5), **kwargs):

    """
    Plot the posterior standard deviation of the :class:`pymc.gp.cov.ExpQuad <pymc.gp.cov.ExpQuad>` covariance kernel lengthscale when 1 through *N* chains are considered. When the posterior has been sufficiently explored, the standard deviation will stabilize; if it has not stabilized, then additional chains should be run.

    To consider chains from multiple traces associated with the same inference model, first combine the traces (saved as NetCDF files) using :py:meth:`combine_traces() <stratmc.data>` in :py:mod:`stratmc.data`.

    .. plot::

        from stratmc.data import load_trace
        from stratmc.plotting import lengthscale_stability

        full_trace = load_trace('examples/example_docs_trace')

        lengthscale_stability(full_trace)

        plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    proxy: str, optional
        Target proxy; only required if more than one proxy was included in the inference.

    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure showing the standard deviation of the covariance kernel lengthscale hyperparameter posterior for 1 through *N* chains.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']

    else:
        proxy = proxies[0]

    gp_ls_variance = calculate_lengthscale_stability(full_trace, proxy = proxy)

    fig = plt.figure(figsize = figsize)
    ax = fig.gca()

    ax.plot(np.arange(len(gp_ls_variance)) + 1, gp_ls_variance, color = 'k')

    ax.set_xlabel('Number of chains', fontsize = fs)
    ax.set_ylabel('Lengthscale standard deviation (%s)' % proxy, fontsize = fs)

    ax.tick_params(labelsize = fs)

    fig.tight_layout()

    return fig

def proxy_signal_stability(full_trace, figsize = (5, 3.5), **kwargs):
    """
    Plot the sum (over all time slices) of the residuals between the median inferred proxy signal when all *N* chains are considered compared to when 1 to *N-1* chains are considered. When the posterior has been sufficiently explored, the residuals will stabilize and approach zero; if they have not stabilized, then additional chains should be run.

    To consider chains from multiple traces associated with the same inference model, first combine the traces (saved as NetCDF files) using :py:meth:`combine_traces() <stratmc.data>` in :py:mod:`stratmc.data`.

    .. plot::

        from stratmc.data import load_trace
        from stratmc.plotting import proxy_signal_stability

        full_trace = load_trace('examples/example_docs_trace')

        proxy_signal_stability(full_trace)

        plt.show()

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    proxy: str, optional
        Target proxy; only required if more than one proxy was included in the inference.


    Returns
    -------
    fig: matplotlib.pyplot.figure
        Figure showing the stability of the proxy signal inference.
    """

    if 'fontsize' in kwargs:
        fs = kwargs['fontsize']
    else:
        fs = 12

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if 'proxy' in kwargs:
        proxy = kwargs['proxy']

    else:
        proxy = proxies[0]

    median_residuals = calculate_proxy_signal_stability(full_trace, proxy = proxy)

    fig = plt.figure(figsize = figsize)
    ax = fig.gca()

    ax.plot(np.arange(median_residuals.shape[0]) + 1, np.sum(median_residuals, axis = 1), color = 'k')

    ax.set_xlabel('Number of chains', fontsize = fs)
    ax.set_ylabel('Sum of median ' + str(proxy) + ' residuals', fontsize = fs)

    ax.tick_params(labelsize = fs)

    fig.tight_layout()

    return fig
