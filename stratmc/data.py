import pickle
import sys
import warnings

import arviz as az
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", ".*The group X_new is not defined in the InferenceData scheme.*")
warnings.filterwarnings("ignore", ".*X_new group is not defined in the InferenceData scheme.*")

def load_data(sample_file, ages_file, proxies = ['d13c'], proxy_sigma_default = 0.1, drop_excluded_samples = False, drop_excluded_ages = True):
    """
    Import and pre-process proxy data and age constraints from .csv files formatted according to the :ref:`Data table formatting <datatable_target>` guidelines. To combine data from different .csv files, load each file separately and then combine the DataFrames with :py:meth:`combine_data() <stratmc.data>`.

    If ``sample_file.csv`` includes multiple proxy observations from the same stratigraphic horizon (for a given proxy), then all measurements marked ``Exclude? = False`` will be combined using :py:meth:`combine_duplicates() <stratmc.data>`.

    Parameters
    ----------
    sample_file: str
        Path to .csv file containing proxy data for all sections (without '.csv` extension).

    ages_file: str
        Path to .csv file containing age constraints for all sections (without '.csv` extension).

    proxies: str or list(str), optional
        Tracer names (must match column headers in ``sample_file.csv``); defaults to 'd13c`.

    proxy_sigma_default: float or dict{float}, optional
        Measurement uncertainty (:math:`1\\sigma`) to use for proxy observations if not specified in ``proxy_std`` column of ``sample_df``. To set a different value for each proxy, pass a dictionary with proxy names as keys. Defaults to 0.1.

    drop_excluded_samples: bool, optional
        Whether to remove samples with ``Exclude? = True`` from the ``sample_df``; defaults to ``False``. If excluded samples are not dropped, their ages will be passively tracked within the inference model (but they will not be considered during the proxy signal reconstruction).

    drop_excluded_ages: bool, optional
        Whether to remove ages with ``Exclude? = True`` from the ``ages_df``; defaults to ``True``.

    Returns
    -------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.
    """

    if type(proxies) == str:
        proxies = list([proxies])

    samples = pd.read_csv(sample_file + '.csv')
    ages = pd.read_csv(ages_file + '.csv')

    samples['section']=samples['section'].apply(str)
    ages['section']=ages['section'].apply(str)

    if 'shared?' not in list(ages.columns):
        ages['shared?'] = False

    if 'name' not in list(ages.columns):
        ages['name'] = np.nan

    if 'distribution_type' not in list(ages.columns):
        ages['distribution_type'] = 'Normal'

    if 'param_1' not in list(ages.columns):
        ages['param_1'] = np.nan

    if 'param_1_name' not in list(ages.columns):
        ages['param_1_name'] = np.nan

    if 'param_2' not in list(ages.columns):
        ages['param_2'] = np.nan

    if 'param_2_name' not in list(ages.columns):
        ages['param_2_name'] = np.nan

    if 'intermediate detrital?' not in list(ages.columns):
        ages['intermediate detrital?'] = False

    if 'intermediate intrusive?' not in list(ages.columns):
        ages['intermediate intrusive?'] = False

    if 'Exclude?' not in list(ages.columns):
        ages['Exclude?'] = False

    if 'Exclude?' not in list(samples.columns):
        samples['Exclude?'] = False

    if ('depth' in list(samples.columns)) or ('depth' in list(ages.columns)):
        sample_df, ages_df = depth_to_height(samples, ages)

    else:
        sample_df = samples
        ages_df = ages

    if drop_excluded_samples:
        sample_df = sample_df[~sample_df['Exclude?']]

    if drop_excluded_ages:
        ages_df = ages_df[~ages_df['Exclude?']]

    # where there's more than 1 measurement for a proxy, combine
    sample_df = combine_duplicates(sample_df, proxies, proxy_sigma_default)

    return sample_df, ages_df

def depth_to_height(sample_df, ages_df):
    """
    Helper function for converting depth in core to height in section.

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    Returns
    -------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections, with depth in core converted to height in section.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections, with depth in core converted to height in section.
    """

    height = {}
    age_height = {}

    sections = np.unique(sample_df['section'])

    for section in sections:
        # if there are depth values, convert to height in section
        if not sample_df[sample_df['section'] == section]['depth'].isnull().all():
            depth_vec = sample_df[sample_df['section'] == section]['depth'].values
            age_depth_vec = ages_df[ages_df['section'] == section]['depth'].values
            all_depths = np.concatenate((sample_df[sample_df['section'] == section]['depth'].values,
                                        ages_df[ages_df['section'] == section]['depth'].values))
            max_depth = np.nanmax(all_depths)
            height[section] = (depth_vec - max_depth) * -1
            age_height[section] = (age_depth_vec - max_depth) * -1
            sample_ind = sample_df.index[sample_df['section'] == section]
            age_ind = ages_df.index[ages_df['section'] == section]
            sample_df.loc[sample_ind, 'height'] = height[section]
            ages_df.loc[age_ind, 'height'] = age_height[section]

    ages_df = ages_df.sort_values(by = ['section', 'height'])
    sample_df = sample_df.sort_values(by = ['section', 'height'])

    return sample_df, ages_df

def clean_data(sample_df, ages_df, proxies, sections):
    """
    Helper function for cleaning sample data before running an inversion. Sets ``Exclude?`` to ``True`` for samples with no relevant proxy observations, removes sections where all samples have been excluded, and drops excluded age constraints.

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    proxies: str or list(str)
        Tracers to include in the inference.

    sections: list(str) or numpy.array(str)
        List of sections to include in the inference (as named in ``sample_df`` and ``ages_df``).

    Returns
    -------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing cleaned proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing cleaned age constraint data for all sections.

    """

    if type(proxies) == str:
        proxies = list([proxies])

    if sample_df is not None:

        # create a copy so it doesn't  modify the original DataFrame
        sample_df = sample_df.copy()

        keep_idx = np.sort(np.unique((np.concatenate([sample_df.index[~np.isnan(sample_df[proxy])] for proxy in proxies]))))

        exclude_idx = list(sample_df.index)

        for idx in keep_idx:
            exclude_idx.remove(idx)

        # if sample has no relevant proxy observations, exclude from inference
        sample_df.loc[exclude_idx, 'Exclude?'] = True

        sample_df = sample_df[sample_df['section'].isin(sections)]

        sample_df = sample_df.sort_values(by = ['section', 'height'])

        sample_df = sample_df.reset_index(inplace = False, drop = True)


    if ages_df is not None:

        ages_df = ages_df.copy()

        ages_df = ages_df[ages_df['section'].isin(sections)]

        ages_df = ages_df[ages_df['Exclude?'] == False]

        ages_df = ages_df.sort_values(by = ['section', 'height'])

        ages_df = ages_df.reset_index(inplace = False, drop = True)

    return sample_df, ages_df

def combine_duplicates(sample_df, proxies, proxy_sigma_default = 0.1):
    """
    Helper function for combining multiple proxy measurements from the same stratigraphic horizon. For each horizon with multiple proxy values, replaces the proxy value with the mean, and replaces the standard deviation with the combined uncertainty (``proxy_std`` values summed in quadrature) for all measurements. The standard deviation of the population of proxy values for each horizon is stored in the ``proxy_population_std`` column of ``sample_df`` (in :py:meth:`build_model() <stratmc.model.build_model>`, the uncertainty of each proxy observation is modeled as the ``proxy_std`` and ``proxy_population_std`` values summed in quadrature).

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    proxies: list(str)
        List of proxies to include in the inference.

    proxy_sigma_default: float or dict{float}, optional
        Measurement uncertainty (:math:`1\\sigma`) to use for proxy observations if not specified in ``proxy_std`` column of ``sample_df``. To set a different value for each proxy, pass a dictionary with proxy names as keys. Defaults to 0.1.

    Returns
    -------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data with duplicates combined.

    """

    sample_df = sample_df.copy()

    if type(proxies) == str:
        proxies = list([proxies])

    if ((type(proxy_sigma_default) == float) or (type(proxy_sigma_default) == int)):
        temp = proxy_sigma_default
        proxy_sigma_default = {}
        for proxy in proxies:
            proxy_sigma_default[proxy] = temp

    for proxy in proxies:
        if proxy + '_std' not in list(sample_df.columns):
            sample_df[proxy + '_std'] = np.nan

        idx = np.isnan(sample_df[proxy + '_std'])
        sample_df.loc[idx, proxy + '_std'] = proxy_sigma_default[proxy]

    # don't consider excluded samples when averaging observations from same height -- remove from dataframe and add back later
    excluded_sample_df = sample_df[sample_df['Exclude?']]
    sample_df = sample_df[~sample_df['Exclude?'].values.astype(bool)]

    excluded_sample_df.reset_index(inplace = True, drop = True)
    sample_df.reset_index(inplace = True, drop = True)

    dup_idx = np.where(sample_df.duplicated(subset = ['section', 'height'], keep = 'first').values)[0]
    dup_idx = list(sample_df.iloc[dup_idx].index)

    duplicate_dicts = []

    for idx in dup_idx:
        if idx in list(sample_df.index):
            duplicate_rows = (sample_df['section'] == sample_df['section'][idx]) & (sample_df['height'] == sample_df['height'][idx])
            duplicate_df = sample_df[duplicate_rows].copy()
            duplicate_sub_idx = list(duplicate_df.index)

            duplicate_dict = {}

            proxy_columns = [proxy for proxy in proxies]
            proxy_std_columns = [proxy + '_std' for proxy in proxies]
            columns = list(sample_df.columns)

            for c in proxy_columns:
                columns.remove(c)
                # replace proxy value with the mean
                duplicate_dict[c] = np.nanmean(duplicate_df[c])
                # standard deviation of the population of proxy values
                duplicate_dict[c + '_population_std'] = np.nanstd(duplicate_df[c])

            for c in proxy_std_columns:
                columns.remove(c)
                # replace the measurement uncertainty with the quadrature uncertainty (of the measurement uncertainties; this does not include the population standard deviation)
                duplicate_dict[c] = np.sqrt(np.sum((duplicate_df[c])**2))

            for c in columns:
                duplicate_dict[c] = duplicate_df.iloc[0][c]

            for key in list(duplicate_dict.keys()):
                duplicate_dict[key] = [duplicate_dict[key]]

            # removes the duplicate samples from sample_df
            sample_df.drop(index = duplicate_sub_idx, inplace = True)

            duplicate_dicts.append(duplicate_dict)

    for duplicate in duplicate_dicts:
        sample_df = pd.concat([sample_df, pd.DataFrame.from_dict(duplicate)], ignore_index = True)

    # put the excluded samples back
    if excluded_sample_df.shape[0] > 0:
        sample_df = pd.concat([sample_df, excluded_sample_df], ignore_index = True)

    # sort and reset indexing
    sample_df.sort_values(by = ['section', 'height'], inplace = True)

    sample_df.reset_index(inplace = True, drop = True)

    return sample_df


def combine_data(dataframes):
    """
    Helper function for merging :class:`pandas.DataFrame` objects containing proxy observations or age constraints. Data are merged using the ``section`` and ``height`` columns.

    Parameters
    ----------
    dataframes: list(pandas.DataFrame)
        List of :class:`pandas.DataFrame` objects to merge.

    Returns
    -------
    merged_data: pandas.DataFrame
        :class:`pandas.DataFrame` containing merged data.
    """

    data = pd.DataFrame(columns = ['section', 'height'])

    for df in dataframes:
        data = data.merge(df,
                          how = 'outer')

    return data

def combine_traces(trace_list):

    """
    Helper function for combining multiple :class:`arviz.InferenceData` objects (saved as NetCDF files) that contain prior and posterior samples for the same inference model (sampled with :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`). The :class:`arviz.InferenceData` objects are concatenated along the ``chain`` dimension such that if two traces with 8 chains each are concatenated, the new combined trace will have 16 chains.

    Parameters
    ----------
    trace_list: list(str)
       List of paths to :class:`arviz.InferenceData` objects (saved as NetCDF files) to be merged.

    Returns
    -------
    combined_trace: arviz.InferenceData
        New :class:`arviz.InferenceData` object containing the prior and posterior draws for all traces in ``trace_list``.
    """

    combined_trace = load_trace(trace_list[0])
    dataset = combined_trace.X_new.copy()
    X_new = combined_trace.X_new.X_new.values

    del combined_trace.X_new
    for path in trace_list[1:]:
        trace = load_trace(path)

        if not np.array_equal(trace.X_new.X_new.values.ravel(), X_new.ravel()):
            sys.exit("Traces have different X_new - check that all inferences were run with the same data and parameters")

        del trace.X_new

        az.concat([combined_trace, trace], dim = 'chain', inplace = True)

    combined_trace.add_groups(dataset)

    return combined_trace

def drop_chains(full_trace, chains):

    """
    Remove a subset of chains from a :class:`arviz.InferenceData` object.

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    chains: list or np.array of int
        Indices of chains to remove from ``full_trace``.

    Returns
    -------
    full_trace_clean: arviz.InferenceData
        Copy of ``full_trace`` without the chains specified in ``chains``.

    """

    all_chains = list(full_trace.posterior.chain.values)

    for chain in chains:
        all_chains.remove(chain)

    full_trace_clean = full_trace.sel(chain = all_chains, inplace = False)

    return full_trace_clean

def thin_trace(full_trace, drop_freq = 2):
    """
    Remove a subset of draws from a :class:`arviz.InferenceData` object. Only applies to groups associated with the posterior (the prior draws will not be affected).

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    drop_freq: int
        Frequency of draw removal. For example, 2 will remove every other draw, while 4 will remove every fourth draw.

    Returns
    -------
    thinned_trace: arviz.InferenceData
        Thinned version of ``full_trace``.

    """
    all_draws = list(full_trace.posterior.draw.values)

    drop_draws = list(full_trace.posterior.draw.values)[::drop_freq]

    for draw in drop_draws:
        all_draws.remove(draw)

    thinned_trace = full_trace.sel(groups = ["posterior", "posterior_predictive", "sample_stats", "log_likelihood"], draw = all_draws, inplace = False)

    return thinned_trace


def save_trace(trace, path):
    """
    Save trace (:class:`arviz.InferenceData` object) as a NetCDF file.

    Parameters
    ----------

    trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`build_model() <stratmc.model>` in :py:mod:`stratmc.model` (the output of :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`).

    path: str
        Location (including the file name, without '.nc` extension) to save ``trace``.

    """

    trace.to_netcdf(path+'.nc', groups=['posterior', 'log_likelihood', 'prior', 'prior_predictive', 'posterior_predictive', 'observed_data', 'sample_stats', 'X_new'])


def save_object(var, path):
    """
    Save variable as a pickle (.pkl) object.

    Parameters
    ----------
    var:
        Variable to be saved.

    path: str
        Location (including the file name, without '.pkl` extension) to save ``var``.

    """

    with open(path+'.pkl', "wb") as buff:
        pickle.dump(var, buff)


def load_trace(path):
    """
    Custom load command for NetCDF file containing a trace (:class:`arviz.InferenceData` object saved with :py:meth:`save_trace() <stratmc.data.save_trace>`).

    Parameters
    ----------
    path: str
        Path to saved NetCDF file (without the '.nc` extension).

    Returns
    -------
    trace: arviz.InferenceData
        Trace saved as NetCDF file.

    """

    trace = az.from_netcdf(path+'.nc')

    return trace

def load_object(path):
    """
    Custom load command for pickle (.pkl) object (variables can be saved as .pkl files with :py:meth:`save_object() <stratmc.data.save_object>`).

    Parameters
    ----------
    path: str
        Path to saved .pkl file (without the '.pkl` extension).

    Returns
    -------
    var:
       Variable saved in ``path``.

    """

    with open(path + '.pkl', "rb") as input_file:
        return pickle.load(input_file)

def accumulation_rate(full_trace, sample_df, ages_df, method = 'all', age_model = 'posterior', include_age_constraints = True, **kwargs):
    """
    Calculate apparent sediment accumulation rate between successive samples (if ``method = 'successive'``) or every possible sample pairing (``method = 'all'``).

    Note that if ``method = 'all'``, rate is returned in mm/year, and duration is returned in years. If ``method = 'successive'``, rate is returned in m/Myr, and duration is returned in Myr. Input data are assumed to have units of meters and millions of years. Used as input to :py:meth:`sadler_plot() <stratmc.plotting>` and :py:meth:`accumulation_rate_stratigraphy() <stratmc.plotting>` in :py:mod:`stratmc.plotting`.

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing all proxy data.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints from all sections.

    method: str, optional
        Whether to calculate accumulation rates between every possible sample pairing ('all`), or between successive samples ('successive`); defaults to 'all`.

    age_model: str, optional
        Whether to calculate accumulation rates using the the posterior or prior age model for each section; defaults to 'posterior`.

    include_age_constraints: bool, optional
        Whether to include radiometric age constraints in accumulation rate calculations; defaults to ``True``.

    sections: list(str) or numpy.array(str), optional
        List of sections to include. Defaults to all sections in ``sample_df``.

    Returns
    -------
    rate_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing sediment accumulation rates and associated durations.

    """


    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["prior"].data_vars.keys()) # posterior
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

    if method == 'all': # in mm/yr
        duration = {}
        rate = {}

        rate_df = pd.DataFrame(columns = ['section', 'duration', 'rate'])

        for section in sections:
            section_df = sample_df[sample_df['section']==section]
            sample_heights = section_df['height'].values * 1000 # convert meters to mm
            age_heights = ages_df['height'][(ages_df['section']==section) & (~ages_df['Exclude?']) & (~ages_df['intermediate detrital?'])  & (~ages_df['intermediate intrusive?'])] * 1000 # convert meters to mm

            duration[section] = []
            rate[section] = []

            # shape (samples x draws)
            if age_model == 'posterior':
                sample_age_post = az.extract(full_trace.posterior)[str(section) + '_ages'].values
                age_constraint_post = az.extract(full_trace.posterior)[str(section) + '_radiometric_age'].values

            elif age_model == 'prior':
                sample_age_post = az.extract(full_trace.prior)[str(section) + '_ages'].values
                age_constraint_post = az.extract(full_trace.prior)[str(section) + '_radiometric_age'].values

            if sample_age_post.shape[0] != len(sample_heights):
                sys.exit(f"Number of data points for {section} does not match the number of data points in the trace.")

            if include_age_constraints:
                comb_heights = np.concatenate([sample_heights, age_heights])

                sort_idx = np.argsort(comb_heights)

                posterior_ages_stacked = np.vstack([sample_age_post, age_constraint_post])

                draws = posterior_ages_stacked.shape[1]

            else:
                posterior_ages_stacked = sample_age_post

                comb_heights = sample_heights

                sort_idx = np.argsort(comb_heights)

                draws = posterior_ages_stacked.shape[1]

            sorted_heights = comb_heights[sort_idx]

            max_idx = len(sorted_heights)

            # for each draw
            for n in np.arange(draws):
                ages = posterior_ages_stacked[sort_idx, n] * 1e6 # put in order, and convert Myr to years
                for i in np.arange(len(sorted_heights)):
                    for j in np.arange(i+1, max_idx): # if at the top sample, returns empty array
                        height_diff = sorted_heights[j] - sorted_heights[i]
                        age_diff = ages[i] - ages[j]
                        duration[section].append(age_diff)
                        rate[section].append(height_diff/age_diff)

            section_rate_df = pd.DataFrame({'section': [section] * len(duration[section]), 'duration': duration[section], 'rate': rate[section]})

            rate_df = pd.concat([rate_df.astype(section_rate_df.dtypes), section_rate_df], ignore_index = True)

    elif method == 'successive': # in meters/Myr
        duration = {}
        rate = {}

        base_height = {}
        top_height = {}
        base_age = {}
        top_age = {}

        rate_df = pd.DataFrame(columns = ['section',  'base_height', 'top_height', 'base_age', 'top_age', 'duration', 'rate'])

        for section in sections:
            base_height[section] = []
            base_age[section] = []
            top_height[section] = []
            top_age[section] = []
            duration[section] = []
            rate[section] = []

            section_df = sample_df[sample_df['section']==section]
            sample_heights = section_df['height'].values
            age_heights = ages_df['height'][(ages_df['section']==section) & (~ages_df['Exclude?']) & (~ages_df['intermediate detrital?'])  & (~ages_df['intermediate intrusive?'])]

            # shape (samples x draws)
            if age_model == 'posterior':
                sample_age_post = az.extract(full_trace.posterior)[str(section) + '_ages'].values
                age_constraint_post = az.extract(full_trace.posterior)[str(section) + '_radiometric_age'].values

            elif age_model == 'prior':
                sample_age_post = az.extract(full_trace.prior)[str(section) + '_ages'].values
                age_constraint_post = az.extract(full_trace.prior)[str(section) + '_radiometric_age'].values


            if sample_age_post.shape[0] != len(sample_heights):
                sys.exit(f"Number of data points for {section} does not match the number of data points in the trace.")

            if include_age_constraints:
                comb_heights = np.concatenate([sample_heights, age_heights])

                sort_idx = np.argsort(comb_heights)

                posterior_ages_stacked = np.vstack([sample_age_post, age_constraint_post])

                draws = posterior_ages_stacked.shape[1]

            else:
                posterior_ages_stacked = sample_age_post

                comb_heights = sample_heights

                sort_idx = np.argsort(comb_heights)

                draws = posterior_ages_stacked.shape[1]

            sorted_heights = comb_heights[sort_idx]

            max_idx = len(sorted_heights) - 1

            # for each draw
            for n in np.arange(draws):
                ages = posterior_ages_stacked[sort_idx, n] # keep in Myr
                for i in np.arange(len(sorted_heights)-1):
                    height_diff = sorted_heights[i+1] - sorted_heights[i]
                    age_diff = ages[i] - ages[i+1]
                    base_age[section].append(ages[i])
                    base_height[section].append(sorted_heights[i])
                    top_age[section].append(ages[i+1])
                    top_height[section].append(sorted_heights[i+1])
                    duration[section].append(age_diff)
                    rate[section].append(height_diff/age_diff)

            section_rate_df = pd.DataFrame({'section': [section] * len(duration[section]),
                                            'base_height': base_height[section],
                                            'top_height': top_height[section],
                                            'base_age': base_age[section],
                                            'top_age': top_age[section],
                                            'duration': duration[section],
                                            'rate': rate[section]})

            rate_df = pd.concat([rate_df.astype(section_rate_df.dtypes), section_rate_df], ignore_index = True)

    return rate_df
