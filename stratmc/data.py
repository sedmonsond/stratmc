import pickle
import sys
import warnings

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", ".*The group X_new is not defined in the InferenceData scheme.*")
warnings.filterwarnings("ignore", ".*X_new group is not defined in the InferenceData scheme.*")

def load_data(sample_file, ages_file, proxies = ['d13c'], proxy_sigma_default = 0.1, drop_excluded_samples = False, drop_excluded_ages = True):
    """
    Import and pre-process proxy data and age constraints from .csv files formatted according to the :ref:`Data table formatting <datatable_target>` guidelines. To combine data from different .csv files, load each file separately and then combine the DataFrames with :py:meth:`combine_data() <stratmc.data>`.

    If ``sample_file.csv`` includes multiple proxy observations from the same stratigraphic horizon (for a given proxy), then all measurements marked ``Exclude? = False`` and ``superposition? = True ``  will be combined using :py:meth:`combine_duplicates() <stratmc.data>`. Samples marked ``superposition? = False`` will remain separate, and their order will be randomized within the inference model.

    Parameters
    ----------
    sample_file: str
        Path to .csv file containing proxy data for all sections (without '.csv` extension).

    ages_file: str
        Path to .csv file containing age constraints for all sections (without '.csv` extension).

    proxies: str or list(str), optional
        proxy names (must match column headers in ``sample_file.csv``); defaults to 'd13c`.

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

    ages['name']=ages['name'].apply(str)

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

    if 'superposition?' not in list(samples.columns):
        samples['superposition?'] = True

    if 'depositional age' not in list(samples.columns):
        samples['depositional age'] = np.nan

    if ('depth' in list(samples.columns)) or ('depth' in list(ages.columns)):
        sample_df, ages_df = depth_to_height(samples, ages)

    else:
        sample_df = samples
        ages_df = ages

    if drop_excluded_samples:
        sample_df = sample_df[~sample_df['Exclude?']]

    if drop_excluded_ages:
        ages_df = ages_df[~ages_df['Exclude?']]


    # where there's more than 1 measurement for a proxy, combine (unless superposition = False)
    sample_df = combine_duplicates(sample_df, proxies, proxy_sigma_default)

    ages_df.sort_values(by = ['section', 'height'], inplace = True)

    ages_df.reset_index(inplace = True, drop = True)

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
        Proxies to include in the inference.

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
    no_superposition_sample_df = sample_df[~sample_df['superposition?']]
    sample_df = sample_df[(~sample_df['Exclude?'].values.astype(bool)) & (sample_df['superposition?'].values.astype(bool))]

    excluded_sample_df.reset_index(inplace = True, drop = True)
    no_superposition_sample_df.reset_index(inplace = True, drop = True)
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

    # put samples w/out superposition information back
    if no_superposition_sample_df.shape[0] > 0:
        sample_df = pd.concat([sample_df, no_superposition_sample_df], ignore_index = True)

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

def upsample(full_trace, downsampled_df, sample_df, ages_df, **kwargs):
    """
    Extend age models calculated using downsampled proxy observations from :py:meth:`downsample() <bayestrat.data>` to the full set of proxy observations.

    .. todo::
        Remove? Shouldn't be necessary since ages for excluded samples can now be tracked w/in the model
    .. todo::
        Check behavior with excluded samples


    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`build_model() <bayestrat.model>` in :py:mod:`bayestrat.model`.

    downsampled_df: pandas.DataFrame
        Downsampled sample DataFrame from ``bayestrat.data.downsample`` (used for the proxy inference associated with ``full_trace``).

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing all proxy data.
    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints from all sections.

    Returns
    -------
    age_model_summary: pandas.DataFrame
        :class:`pandas.DataFrame` containing sample age summary statistics (mean, standard deviation, median, and 68% and 95% confidence intervals) for each sample.
    """

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df['section'])

    # get list of proxies included in model from full_trace
    variables = [
            l
            for l in list(full_trace["posterior"].data_vars.keys())
            if f"{'gp_ls_'}" in l
            ]

    proxies = []
    for var in variables:
        proxies.append(var[6:])

    if type(proxies) == str:
        proxies = list([proxies])

    keep_idx = np.sort(np.unique((np.concatenate([downsampled_df.index[~np.isnan(downsampled_df[proxy])] for proxy in proxies]))))

    downsampled_df = downsampled_df.loc[keep_idx]

    downsampled_df = downsampled_df.sort_values(by = ['section', 'height'])

    keep_idx_all = np.sort(np.unique((np.concatenate([sample_df.index[~np.isnan(sample_df[proxy])] for proxy in proxies]))))

    sample_df = sample_df.loc[keep_idx_all]

    sample_df = sample_df.sort_values(by = ['section', 'height'])

    interp_df = pd.DataFrame(columns = list(sample_df.columns) + ['interp'])

    for section in sections:
        # height of samples included in inference
        downsampled_section_df = downsampled_df[downsampled_df['section']==section]
        downsampled_heights = np.concatenate([downsampled_section_df[~np.isnan(downsampled_section_df[proxy])]['height'].values for proxy in proxies])

        downsampled_heights = np.sort(np.unique(downsampled_heights))

        # all sample + age constraint heights
        section_df = sample_df[sample_df['section']==section]
        sample_heights = np.concatenate([section_df[~np.isnan(section_df[proxy])]['height'].values for proxy in proxies])

        sample_heights = np.unique(sample_heights)
        sample_heights = np.sort(sample_heights)

        # heights of radiometric age constraints
        sample_ages_df = ages_df[ages_df['section']==section]
        age_heights = sample_ages_df['height'].values

        # heigts at which to interpolate age models
        interp_heights = [h for h in sample_heights if h not in downsampled_heights]

        interp_section_df = pd.DataFrame(columns = list(sample_df.columns) + ['interp'])
        for h in interp_heights:
            idx = sample_df[sample_df['height']==h].index.tolist()
            interp_section_df = pd.concat([interp_section_df, sample_df.loc[idx]])

        interp_section_df['interp'] = 'y'
        interp_section_df.reset_index(inplace = True, drop = True)

        # sample age posterior - shape (samples x draws)
        sample_age_post = az.extract(full_trace.posterior)[str(section) + '_ages'].values

        age_constraint_post = az.extract(full_trace.posterior)[str(section) + '_radiometric_age'].values

        if sample_age_post.shape[0] != len(downsampled_heights):
            sys.exit(f"Number of data points for {section} does not match the number of data points in the trace. Check that input data and list of proxies match.")

        if age_constraint_post.shape[0] != len(age_heights):
            sys.exit(f"Number of data points for {section} does not match the number of data points in the trace. Check that input data and list of proxies match.")


        # combine position data for all samples + age constraints included in the inference
        all_heights = np.concatenate([downsampled_heights, age_heights])
        sorted_idx = np.argsort(all_heights)
        all_heights_sort = all_heights[sorted_idx]

        # construct age and height vectors for the current draw using posteriors for 1) samples in section, and 2) age constraints
        for i in np.arange(sample_age_post.shape[1]):
            sample_age_vec = sample_age_post[:, i]
            constraint_age_vec = age_constraint_post[:, i]
            age_vec = np.concatenate([sample_age_vec, constraint_age_vec])
            age_vec_sort = age_vec[sorted_idx]

            # interpolate - x is height (must be strictly increasing), y is age
            interp_age = np.interp(interp_heights, all_heights_sort, age_vec_sort)
            interp_age = np.asarray(interp_age).reshape(len(interp_heights), 1)

            # rows = samples, columns = draws
            if i == 0:
                age_paths = interp_age

            else:
                age_paths = np.hstack((age_paths, interp_age))


        downsampled_section_df['interp'] = 'n'
        downsampled_section_df['age_draws'] = np.nan
        downsampled_section_df['age_draws'] = downsampled_section_df['age_draws'].astype(object)

        interp_section_df['age_draws'] = np.nan
        interp_section_df['age_draws'] = interp_section_df['age_draws'].astype(object)
        for i in interp_section_df.index.tolist():
            interp_section_df['age_draws'].loc[i] = age_paths[i, :]

        downsampled_section_df.reset_index(inplace = True, drop = True)
        for i in downsampled_section_df.index.tolist():
            downsampled_section_df['age_draws'].loc[i] = sample_age_post[i, :]

        interp_df = pd.concat([interp_df, downsampled_section_df, interp_section_df])


        interp_df.sort_values(by = ['section', 'height'], inplace = True)
        interp_df.reset_index(inplace = True, drop = True)
        interp_df['mle'] = np.nan
        interp_df['2.5'] = np.nan
        interp_df['16'] = np.nan
        interp_df['50'] = np.nan
        interp_df['84'] = np.nan
        interp_df['97.5'] = np.nan

        for i in interp_df.index.tolist():
            current_ages = interp_df['age_draws'].loc[i]

            # mle
            dy = np.linspace(np.min(current_ages), np.max(current_ages), 2000)
            max_like = dy[np.argmax(gaussian_kde(current_ages, bw_method = 1)(dy))]
            interp_df['mle'].loc[i] = max_like

            # median
            interp_df['50'].loc[i] = np.percentile(current_ages, 50)

            # 2.5%
            interp_df['2.5'].loc[i] = np.percentile(current_ages, 2.5)

            # 16%
            interp_df['16'].loc[i] = np.percentile(current_ages, 16)

            # 84%
            interp_df['84'].loc[i] = np.percentile(current_ages, 84)

            # 97.5%
            interp_df['97.5'].loc[i] = np.percentile(current_ages, 97.5)

    return interp_df

def downsample_kmeans(sample_df, ages_df, method = 'target_cluster_std', proxy = 'd13c', nearest_point = True, **kwargs):
    """
    Downsample proxy observations using k-means clustering. Downsampling is performed on each section 'segment' between pairs of successive age constraints. To downsample multiple proxies, separately downsample each proxy and then merge using :py:meth:`combine_data() <bayestrat.data>`.

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    proxy: str, optional
        Proxy to downsample. Defaults to 'd13c`.

    sections: list(str) or numpy.array(str), optional
        List of sections to downsample. Defaults to all sections in ``sample_df``.

    method: str, optional
        Downsampling method ('target_cluster_std', 'resolution', 'n_clusters', or 'keep_fraction'). Defaults to ``target_cluster_std``, which increases the number of clusters until the within-cluster standard deviation of the proxy values is less than or equal to ``target_cluster_std``.

    target_cluster_std: float or dict{float}, optional
        Target within-cluster standard deviation; used if ``method = 'target_cluster_std'. Defaults to 0.5.

    cluster_std_method: str
        Whether to select the number of clusters such that all clusters have a standard deviation less than or equal to ``target_cluster_std`` ('all`), or such that the mean of the within-cluster standard deviations is less than or equal to ``target_cluster_std`` ('mean`). Defaults to 'all`.

    target_res: float or dict{float}, optional
        Target resolution (vertical distance between samples, in meters). Used if ``method = 'resolution'``; defaults to 5. If a float is passed, uses the same value for every section; to use a different resolution for each section, pass a dictionary with section names as keys.

    n_clusters: int or dict{int}, optional
        Number of clusters. Used if ``method = 'n_clusters'``; defaults to 10. If a float is passed, uses the same value for every segment. To use a different resolution for each section, pass a dictionary with section names as keys. To use a different value for each segment within a given section, set its dictionary entry equal to a dictionary with segment numbers (e.g., 0 for the lowermost segment) as keys.

    keep_fraction: float or dict{float}, optional
        Number of clusters = (numer of samples $\times$ keep_fraction). Used if ``method = 'keep_fraction'``; defaults to 0.5. If a float is passed, uses the same value for every section. To use a different resolution for each section, pass a dictionary with section names as keys.

    min_clusters_per_interval: int or dict{int}, optional
        Minimum number of clusters per interval; to use a different value for each section, pass a dictionary with section names as keys. Defaults to 1.

    nearest_point: boolean, optional
        For each cluster, keep the data point closest to the cluster center (with its measurement uncertainty) and exclude all other observations; defaults to ``True``. If ``False``, instead uses the cluster center (which does not necessarily correspond to a real data point) and excludes the original observations, with 'proxy_std` equal to the population standard deviation of the proxy observations assigned to the cluster.

    Returns
    -------
    downsampled_data: pandas.DataFrame
        :class:`pandas.DataFrame` containing downsampled proxy data. All samples are still included in the DataFrame, but samples that were excluded during downsampling are marked ``Exclude? = True``.

    """

    sample_df_downsampled = sample_df.copy()

    if 'sections' in kwargs:
            sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df_downsampled['section'])

    if method == 'resolution':
        n_clusters = {}
        if 'target_res' in kwargs:
            target_res = kwargs['target_res']
            temp_res = target_res
            if (type(temp_res) == float) or (type(temp_res) == int):
                target_res = {}
                for section in sections:
                    target_res[section] = temp_res
        else:
            target_res = {}
            for section in sections:
                target_res[section] = 5

    elif method == 'n_clusters':
        if 'n_clusters' in kwargs:
            n_clusters = kwargs['n_clusters']
        else:
            n_clusters = {}
            for section in sections:
                n_clusters[section] = 10

    elif method == 'keep_fraction':
        if 'keep_fraction' in kwargs:
            keep_fraction = kwargs['keep_fraction']
            if type(keep_fraction) == float:
                temp = keep_fraction
                keep_fraction = {}
                for section in sections:
                    keep_fraction[section] = temp

        else:
            keep_fraction = {}
            for section in sections:
                keep_fraction[section] = 0.25

    elif method == 'target_cluster_std':
        if 'target_cluster_std' in kwargs:
            target_cluster_std = kwargs['target_cluster_std']
            if type(target_cluster_std) == float:
                temp = target_cluster_std
                target_cluster_std = {}
                for section in sections:
                    target_cluster_std[section] = temp

        else:
            target_cluster_std = {}
            for section in sections:
                target_cluster_std[section] = 0.5

        if 'cluster_std_method' in kwargs:
            cluster_std_method = kwargs['cluster_std_method']
            if type(cluster_std_method) == float:
                temp = cluster_std_method
                cluster_std_method = {}
                for section in sections:
                    cluster_std_method[section] = temp

        else:
            cluster_std_method = {}
            for section in sections:
                cluster_std_method[section] = 'all'

    if 'min_clusters_per_interval' in kwargs:
        min_clusters_per_interval = kwargs['min_clusters_per_interval']
        if type(min_clusters_per_interval) == float:
                temp = min_clusters_per_interval
                min_clusters_per_interval = {}
                for section in sections:
                    min_clusters_per_interval[section] = temp
    else:
        min_clusters_per_interval = {}
        for section in sections:
            min_clusters_per_interval[section] = 1


    downsampled_df = {}
    # proxy_nan_df = {}

    # excluded samples -- put these back into the final DataFrame as-is
    excluded_df =  sample_df_downsampled[sample_df_downsampled['Exclude?']]

    # keep track of samples that aren't marked as exclude, but that don't have any data for the proxy to be downsampled (these should be included in the final dataframe as-is)
    # proxy_nan_df[proxy] = sample_df_downsampled[((~sample_df_downsampled['Exclude?']) & (np.isnan(sample_df_downsampled[proxy]))]

    downsampled_df[proxy] = pd.DataFrame(columns = ['section', 'height', proxy, proxy + '_std', 'superposition?', 'Exclude?', 'Depositional Environment'])

    for section in tqdm(sections):
        print(f'Downsampling {section}')
        section_df = sample_df_downsampled[(sample_df_downsampled['section']==section) & (~sample_df_downsampled['Exclude?'].astype(bool))].dropna(subset = proxy)
        section_ages_df = ages_df[(ages_df['section']==section)  & (~ages_df['depositional?'])]
        heights = section_df['height'].values
        proxy_vec = section_df[proxy].values
        age_heights = section_ages_df['height'].values

        section_unique_dep_env = section_df['Depositional Environment'].unique()
        section_dep_env = section_df['Depositional Environment'].values
        section_superposition = section_df['superposition?'].values
        section_dep_ages = section_df['depositional age'].values
        section_unique_dep_ages = list(section_df['depositional age'].astype(str).unique())

        if 'nan' in section_unique_dep_ages:
            section_unique_dep_ages.remove('nan')

        centers = {}
        closest_df_idx = []
        cluster_center_idx = []
        no_downsample_idx = []
        center_std = {}

        intervals = []

        # create a list of interval boundary heights: 1) age constraint, 2) change in 'superposition?' boolean, 3) change in depositional environment
        interval_boundary_heights = list(age_heights)

        # if there's a change in superposition boolean within the interval, need to split into a separate interval (w/ same lower and upper bound -- just add the height twice)
        if not (all(section_superposition)) or (all(~(section_superposition.astype(bool)))):
            # grab heights of samples without superposition
            super_heights = np.unique(section_df[~section_df['superposition?'].astype(bool)]['height'])

            for h in super_heights:
                if h not in interval_boundary_heights:
                    # append the height
                    interval_boundary_heights += [h]
                    top_h_idx = np.where(section_df['height'] == h)[0][-1]

                # bound with height of overlying sample, if not already done or at top of section
                if (h != np.max(heights)):
                    if heights[top_h_idx + 1] not in super_heights:
                        interval_boundary_heights.append(heights[top_h_idx + 1])

        # add boundaries around groups of samples with the same depositional age
        for dep_age in section_unique_dep_ages:
            # print(f'splitting depositional ages for section {section}')
            dep_age_idx = np.where(section_dep_ages == dep_age)[0]

            # assuming all the ages are in 1 chunk, add the base and the overlying sample
            if all(np.diff(dep_age_idx) == 1):

                # only add base if chunk isn't at base of section
                if dep_age_idx[0] != 0:
                    interval_boundary_heights.append(heights[dep_age_idx[0]])

                # only add overlying sample if not at top of section (already bounded by another age constraint)
                if dep_age_idx[-1] != len(heights) - 1:
                    interval_boundary_heights.append(heights[dep_age_idx[-1] + 1])

            else:
                print(f'samples with depositional age {dep_age} in section {section} are not in a continuous chunk - check that depositional age assignment is correct')
                if (len(dep_age_idx) > 1):
                        switch_idx = np.where(np.diff(dep_age_idx) != 1)[0]

                        interval_boundary_heights += list(heights[dep_age_idx[switch_idx] + 1])

                        # add boundary above the uppermost chunk, unless it's the top of the secion  (in which case there should already be an age constraint)
                        if dep_age_idx[-1] != len(heights) - 1:
                            interval_boundary_heights += list([heights[dep_age_idx[-1] + 1]])


        # add boundaries between different depositional environments
        if len(section_unique_dep_env) > 1:
            for env in section_unique_dep_env:
                env_idx = np.where(section_dep_env == env)[0]

                # add base of lowermost group, unless we're at the bottom of the section
                if env_idx[0] != 0:
                    interval_boundary_heights.append(heights[env_idx[0]])

                # if all samples from this environment are in 1 chunk, just add the top boundary
                if (len(env_idx)) >= 1 and (all(np.diff(env_idx) == 1)):
                    # don't add to list if chunk is at top of the section (already bounded by an age constraint)
                    if (env_idx[-1] != len(heights) - 1):
                        interval_boundary_heights.append(heights[env_idx[-1] + 1])

                else:
                    # if there's more than one sample from this environment (scenario with only 1 is covered above)
                    if (len(env_idx) > 1):
                        switch_idx = np.where(np.diff(env_idx) != 1)[0]

                        interval_boundary_heights += list(heights[env_idx[switch_idx] + 1])

                        # add boundary above the uppermost chunk, unless it's the top of the secion  (in which case there should already be an age constraint)
                        if env_idx[-1] != len(heights) - 1:
                            interval_boundary_heights += list([heights[env_idx[-1] + 1]])

            # # if we added the lowermost sample, remove it
            # if heights[0] in interval_boundary_heights:
            #     interval_boundary_heights.remove(heights[0])

        # sort interval boundaries, and get rid of any duplicate boundaries
        interval_boundary_heights = np.sort(np.unique(interval_boundary_heights))

        for interval, boundary_height in enumerate(interval_boundary_heights[:-1]):
            above = section_df['height']>=boundary_height
            below = section_df['height']<interval_boundary_heights[interval+1]
            interval_df = section_df[above & below]

            interval_samples = interval_df[proxy].values

            if len(interval_df['Depositional Environment'].unique()) > 1:
                print(f'error - multiple depositional environments in same interval in section {section}')
                print(interval_df['Depositional Environment'].unique())
                print(boundary_height, interval_boundary_heights[interval + 1])

            if len(interval_df['depositional age'].unique()) > 1:
                print(f'error - multiple depositional ages in same interval in section {section}')
                print(interval_df['depositional age'].unique())
                print(boundary_height, interval_boundary_heights[interval + 1])

            interval_heights = interval_df['height'].values

            if len(interval_heights) > min_clusters_per_interval[section]:

                intervals.append(interval)

                interval_dep_env = interval_df['Depositional Environment'].unique()[0]

                # TODO: put this into the dataframe after clustering
                interval_dep_age = interval_df['depositional age'].unique()[0]


                if method == 'resolution':
                    if len(np.diff(interval_heights) > 0):
                        interval_res = np.mean(np.diff(interval_heights))
                        scale = target_res[section]/interval_res
                        if (scale > 1) and (len(interval_heights) >= min_clusters_per_interval[section]):
                            current_n_clusters = round(len(interval_samples)/scale)
                            if current_n_clusters > 0:
                                go = True
                            else:
                                go = False
                        else:
                            go = False

                    else:
                        go = False

                elif method == 'n_clusters':
                    section_n_clusters = n_clusters[section]
                    if type(section_n_clusters) == dict:
                        current_n_clusters = section_n_clusters[interval]
                    else:
                        current_n_clusters = n_clusters[section]
                    if (current_n_clusters < len(interval_heights)) and (len(interval_heights) >= min_clusters_per_interval[section]) and (current_n_clusters > 0):
                        go = True

                    else:
                        go = False

                elif method == 'keep_fraction':
                    current_n_clusters = np.round(keep_fraction[section] * len(interval_heights)).astype(int)
                    if (current_n_clusters < len(interval_heights)) and (len(interval_heights) >= min_clusters_per_interval[section]) and (current_n_clusters > 0):
                        go = True
                    else:
                        go = False

                elif method == 'target_cluster_std':
                    section_target_cluster_std = target_cluster_std[section]
                    if (len(interval_heights) >= min_clusters_per_interval[section]):
                        go = True
                    else:
                        go = False

                if go and (method != 'target_cluster_std'):
                    X = [[x, y] for x, y in zip(interval_df['height'], interval_df[proxy])]

                    # init kmeans classifier
                    km = KMeans(n_clusters=current_n_clusters, random_state=0,  init='k-means++')

                    # assign a cluster to each example
                    yhat = km.fit_predict(X)

                    # retrieve unique clusters
                    clusters = np.unique(yhat)

                    centers[interval] = km.cluster_centers_
                    center_std[interval] = []
                    delete_centers = []

                    for cluster in clusters:
                        row_ix = np.asarray(np.where(yhat == cluster))[0]
                        X = np.asarray(X)
                        if row_ix.shape[0] > 1:
                            # TODO: should we just use the sample w/ the proxy value closest to the center proxy value (disregarding the height)?
                            # calculate distance between cluster center and each data point in the cluster
                            current_center = centers[interval][cluster]
                            distance = np.zeros(row_ix.shape[0]) * np.nan
                            for i in np.arange(len(row_ix)):
                                # this version calculates distance from cluster center using both heights and proxy values
                                #distance[i] = np.linalg.norm(current_center - X[row_ix[i]])

                                # this version only uses proxy values to calculate distance from cluster center
                                distance[i] = np.abs(current_center[1] - X[row_ix[i]][1])

                            closest_idx = row_ix[np.argmin(distance)]
                            closest_df_idx.append(interval_df.index.tolist()[closest_idx])
                            center_std[interval].append(np.std(X[row_ix, 0]))
                        else:
                            delete_centers.append(cluster)
                            closest_df_idx.append(interval_df.index.tolist()[row_ix[0]])
                            # if the cluster only contains 1 data point, we'll keep it regardless of whether nearest_point is True or False
                            cluster_center_idx.append(interval_df.index.tolist()[row_ix[0]])

                    if len(delete_centers) > 0:
                        center_std[interval] = np.delete(center_std[interval], delete_centers, axis=0)
                        centers[interval] = np.delete(centers[interval], delete_centers, axis=0)

                elif go and (method == 'target_cluster_std'):
                    X = [[x, y] for x, y in zip(interval_df['height'], interval_df[proxy])]

                    # init kmeans classifier
                    current_n_clusters = min_clusters_per_interval[section]
                    # np.nanstd(interval_df[proxy].values)
                    current_cluster_std = [section_target_cluster_std + 1] * min_clusters_per_interval[section]

                    while any(np.array(current_cluster_std) > section_target_cluster_std):
                        current_n_clusters += 1

                        km = KMeans(n_clusters=current_n_clusters, init = 'k-means++', random_state=0)

                        yhat = km.fit_predict(X)

                        # retrieve unique clusters
                        clusters = np.unique(yhat)

                        centers[interval] = km.cluster_centers_
                        center_std_temp = []
                        delete_centers = []

                        for cluster in clusters:
                            row_ix = np.asarray(np.where(yhat == cluster))[0]
                            X = np.asarray(X)

                            if len(row_ix) > 1:
                                center_std_temp.append(np.nanstd(X[row_ix, 1]))

                        if cluster_std_method[section] == 'mean':
                            current_cluster_std = [np.nanmean(center_std_temp)]
                        elif cluster_std_method[section] == 'all':
                            current_cluster_std = np.array(center_std_temp)

                    center_std[interval] = []
                    delete_centers = []

                    for cluster in clusters:
                        row_ix = np.asarray(np.where(yhat == cluster))[0]
                        if row_ix.shape[0] > 1:
                            # TODO: decide if we should just use the sample w/ the proxy value closest to the center proxy value (disregarding the height
                            # calculate distance between cluster center and each data point in the cluster
                            current_center = centers[interval][cluster]
                            distance = np.zeros(row_ix.shape[0]) * np.nan
                            for i in np.arange(len(row_ix)):
                                # this version calculates distance from cluster center using both heights and proxy values
                                # distance[i] = np.linalg.norm(current_center - X[row_ix[i]])

                                # this version only uses proxy values to calculate distance from cluster center
                                distance[i] = np.abs(current_center[1] - X[row_ix[i]][1])

                            closest_idx = row_ix[np.argmin(distance)]
                            closest_df_idx.append(interval_df.index.tolist()[closest_idx])
                            # standard deviation of the proxy values within this cluster
                            center_std[interval].append(np.std(X[row_ix, 1]))
                        else:
                            # if only 1 data point in the cluster, don't need to add it to sample_df if nearest_point = False
                            delete_centers.append(cluster)
                            center_std[interval].append(np.nan)
                            closest_df_idx.append(interval_df.index.tolist()[row_ix[0]])
                            # if the cluster only contains 1 data point, we'll keep it regardless of whether nearest_point is True or False
                            cluster_center_idx.append(interval_df.index.tolist()[row_ix[0]])

                    if len(delete_centers) > 0:
                        centers[interval] = np.delete(centers[interval], delete_centers, axis=0)
                        center_std[interval] = np.delete(center_std[interval], delete_centers, axis=0)

                # if go = False (don't meet the requirements to downsample)
                else:
                    # if the interval doesn't contain enough data points to downsample, we'll keep it regardless of whether nearest_point is True or False
                    no_downsample_idx.append(interval_df.index.tolist())

        if not nearest_point:
            for interval in intervals:
                if interval == intervals[0]:
                    section_centers = np.asarray(centers[interval])
                    section_center_std = np.asarray(center_std[interval])
                else:
                    section_centers = np.concatenate([section_centers, np.asarray(centers[interval])])
                    section_center_std =  np.concatenate([section_center_std, np.asarray(center_std[interval])])

        if nearest_point:
            # mark the samples we're keeping as Exclude? = False, and the rest as Exclude? = True
            # don't change samples that didn't have data for the downsampled proxy anyway
            sample_df_downsampled['Exclude?'][(sample_df_downsampled['section'] == section) & (~np.isnan(sample_df_downsampled[proxy]))] = True
            sample_df_downsampled['Exclude?'].loc[closest_df_idx] = False
            sample_df_downsampled['Exclude?'].loc[no_downsample_idx] = False

        else:
            # mark all the samples in section as Exclude? = True, unless they don't have data for the downsampled proxy
            sample_df_downsampled['Exclude?'][(sample_df_downsampled['section'] == section) & (~np.isnan(sample_df_downsampled[proxy]))] = True
            sample_df_downsampled['Exclude?'].loc[no_downsample_idx] = False
            sample_df_downsampled['Exclude?'].loc[cluster_center_idx] = False # this only applies to clusters that only contain 1 data point


            # if depositional age is present, save in dataframe (if not, set to nan)
            if str(interval_dep_age) != 'nan':
                dep_age_vec = [interval_dep_env] * section_centers.shape[0]
            else:
                dep_age_vec = [np.nan] * section_centers.shape[0]

            # add cluster centers to DataFrame
            downsampled_section_df = pd.DataFrame({'section': [section] * section_centers.shape[0],
                                               'height': section_centers[:, 0],
                                               proxy: section_centers[:, 1],
                                               proxy + '_std': section_center_std,
                                               'superposition?': [True] * section_centers.shape[0],
                                               'depositional age': dep_age_vec,
                                                'Exclude?': [False] * section_centers.shape[0],
                                                'Depositional Environment': [interval_dep_env] * section_centers.shape[0]})

            sample_df_downsampled = pd.concat([sample_df_downsampled, downsampled_section_df], ignore_index = True)

    sample_df_downsampled.sort_values(by = ['section', 'height'], inplace = True)

    sample_df_downsampled['Exclude?'] = sample_df_downsampled['Exclude?'].astype(bool)

    sample_df_downsampled.reset_index(inplace = True, drop = True)


    return sample_df_downsampled

def downsample(sample_df, ages_df, pvalue_thresh = 0.05, proxy = 'd13c', nearest_point = False, **kwargs):
    """
    Downsample proxy observations

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    proxy: str, optional
        Proxy to downsample. Defaults to 'd13c`.

    sections: list(str) or numpy.array(str), optional
        List of sections to downsample. Defaults to all sections in ``sample_df``.

    pvalue_thresh: float, optional

    target_cluster_std: float or dict{float}, optional
        Target within-cluster standard deviation; used if ``method = 'target_cluster_std'. Defaults to 0.5.

    cluster_std_method: str
        Whether to select the number of clusters such that all clusters have a standard deviation less than or equal to ``target_cluster_std`` ('all`), or such that the mean of the within-cluster standard deviations is less than or equal to ``target_cluster_std`` ('mean`). Defaults to 'all`.

    target_res: float or dict{float}, optional
        Target resolution (vertical distance between samples, in meters). Used if ``method = 'resolution'``; defaults to 5. If a float is passed, uses the same value for every section; to use a different resolution for each section, pass a dictionary with section names as keys.

    n_clusters: int or dict{int}, optional
        Number of clusters. Used if ``method = 'n_clusters'``; defaults to 10. If a float is passed, uses the same value for every segment. To use a different resolution for each section, pass a dictionary with section names as keys. To use a different value for each segment within a given section, set its dictionary entry equal to a dictionary with segment numbers (e.g., 0 for the lowermost segment) as keys.

    keep_fraction: float or dict{float}, optional
        Number of clusters = (numer of samples $\times$ keep_fraction). Used if ``method = 'keep_fraction'``; defaults to 0.5. If a float is passed, uses the same value for every section. To use a different resolution for each section, pass a dictionary with section names as keys.

    min_clusters_per_interval: int or dict{int}, optional
        Minimum number of clusters per interval; to use a different value for each section, pass a dictionary with section names as keys. Defaults to 1.

    nearest_point: boolean, optional
        For each cluster, keep the data point closest to the cluster center (with its measurement uncertainty) and exclude all other observations; defaults to ``True``. If ``False``, instead uses the cluster center (which does not necessarily correspond to a real data point) and excludes the original observations, with 'proxy_std` equal to the population standard deviation of the proxy observations assigned to the cluster.

    Returns
    -------
    downsampled_data: pandas.DataFrame
        :class:`pandas.DataFrame` containing downsampled proxy data. All samples are still included in the DataFrame, but samples that were excluded during downsampling are marked ``Exclude? = True``.

    """

    sample_df_downsampled = sample_df.copy()
    sample_df_downsampled['cluster'] = np.nan

    if 'sections' in kwargs:
            sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df_downsampled['section'])


    downsampled_df = {}
    # proxy_nan_df = {}

    # excluded samples -- put these back into the final DataFrame as-is
    excluded_df =  sample_df_downsampled[sample_df_downsampled['Exclude?']]

    # keep track of samples that aren't marked as exclude, but that don't have any data for the proxy to be downsampled (these should be included in the final dataframe as-is)
    # proxy_nan_df[proxy] = sample_df_downsampled[(sample_df_downsampled['section']==section) & (~sample_df_downsampled['Exclude?']) & (np.isnan(sample_df_downsampled[proxy]))]

    downsampled_df[proxy] = pd.DataFrame(columns = ['section', 'height', proxy, proxy + '_std', 'superposition?', 'Exclude?', 'Depositional Environment'])

    for section in tqdm(sections):
        print(f'Downsampling {section}')

        section_cluster_number = 0

        section_df = sample_df_downsampled[(sample_df_downsampled['section']==section) & (~sample_df_downsampled['Exclude?'].astype(bool))].dropna(subset = proxy)
        section_ages_df = ages_df[(ages_df['section']==section)  & (~ages_df['depositional?'])]
        heights = section_df['height'].values
        proxy_vec = section_df[proxy].values
        age_heights = section_ages_df['height'].values

        section_unique_dep_env = section_df['Depositional Environment'].unique()
        section_dep_env = section_df['Depositional Environment'].values
        section_superposition = section_df['superposition?'].values
        section_dep_ages = section_df['depositional age'].values
        section_unique_dep_ages = list(section_df['depositional age'].astype(str).unique())

        if 'nan' in section_unique_dep_ages:
            section_unique_dep_ages.remove('nan')


        centers = {}
        clusters = {}
        closest_df_idx = []
        cluster_center_idx = []
        no_downsample_idx = []
        cluster_centers = {}
        cluster_std = {}

        intervals = []
        cluster_center_intervals = []
        cluster_numbers = []

        # create a list of interval boundary heights: 1) age constraint, 2) change in 'superposition?' boolean, 3) change in depositional environment
        interval_boundary_heights = list(age_heights)

        # if there's a change in superposition boolean within the interval, need to split into a separate interval (w/ same lower and upper bound -- just add the height twice)
        if not (all(section_superposition)) or (all(~section_superposition)):
            # grab heights of samples without superposition
            super_heights = np.unique(section_df[~section_df['superposition?']]['height'])

            for h in super_heights:
                if h not in interval_boundary_heights:
                    # append the height
                    interval_boundary_heights += [h]
                    top_h_idx = np.where(section_df['height'] == h)[0][-1]

                # bound with height of overlying sample, if not already done or at top of section
                if (h != np.max(heights)):
                    if heights[top_h_idx + 1] not in super_heights:
                        interval_boundary_heights.append(heights[top_h_idx + 1])

        # add boundaries around groups of samples with the same depositional age
        for dep_age in section_unique_dep_ages:
            # print(f'splitting depositional ages for section {section}')
            dep_age_idx = np.where(section_dep_ages == dep_age)[0]

            # assuming all the ages are in 1 chunk, add the base and the overlying sample
            if all(np.diff(dep_age_idx) == 1):

                # only add base if chunk isn't at base of section
                if dep_age_idx[0] != 0:
                    interval_boundary_heights.append(heights[dep_age_idx[0]])

                # only add overlying sample if not at top of section (already bounded by another age constraint)
                if dep_age_idx[-1] != len(heights) - 1:
                    interval_boundary_heights.append(heights[dep_age_idx[-1] + 1])

            else:
                print(f'samples with depositional age {dep_age} in section {section} are not in a continuous chunk - check that depositional age assignment is correct')
                if (len(dep_age_idx) > 1):
                        switch_idx = np.where(np.diff(dep_age_idx) != 1)[0]

                        interval_boundary_heights += list(heights[dep_age_idx[switch_idx] + 1])

                        # add boundary above the uppermost chunk, unless it's the top of the secion  (in which case there should already be an age constraint)
                        if dep_age_idx[-1] != len(heights) - 1:
                            interval_boundary_heights += list([heights[dep_age_idx[-1] + 1]])


        # add boundaries between different depositional environments
        if len(section_unique_dep_env) > 1:
            for env in section_unique_dep_env:
                env_idx = np.where(section_dep_env == env)[0]

                # add base of lowermost group, unless we're at the bottom of the section
                if env_idx[0] != 0:
                    interval_boundary_heights.append(heights[env_idx[0]])

                # if all samples from this environment are in 1 chunk, just add the top boundary
                if (len(env_idx)) >= 1 and (all(np.diff(env_idx) == 1)):
                    # don't add to list if chunk is at top of the section (already bounded by an age constraint)
                    if (env_idx[-1] != len(heights) - 1):
                        interval_boundary_heights.append(heights[env_idx[-1] + 1])

                else:
                    # if there's more than one sample from this environment (scenario with only 1 is covered above)
                    if (len(env_idx) > 1):
                        switch_idx = np.where(np.diff(env_idx) != 1)[0]

                        interval_boundary_heights += list(heights[env_idx[switch_idx] + 1])

                        # add boundary above the uppermost chunk, unless it's the top of the secion  (in which case there should already be an age constraint)
                        if env_idx[-1] != len(heights) - 1:
                            interval_boundary_heights += list([heights[env_idx[-1] + 1]])

            # # if we added the lowermost sample, remove it
            # if heights[0] in interval_boundary_heights:
            #     interval_boundary_heights.remove(heights[0])

        # sort interval boundaries, and get rid of any duplicate boundaries
        interval_boundary_heights = np.sort(np.unique(interval_boundary_heights))

        for interval, boundary_height in enumerate(interval_boundary_heights[:-1]):
            above = section_df['height']>=boundary_height
            below = section_df['height']<interval_boundary_heights[interval+1]
            interval_df = section_df[above & below]

            interval_proxy = interval_df[proxy]#.values
            interval_heights = interval_df['height']#.values

            if len(interval_heights) > 0:

                cluster_centers[interval] = []
                cluster_std[interval] = []

                # clusters[interval] = np.ones(len(interval_proxy)) * np.nan
                clusters[interval] = pd.Series(np.ones(len(interval_proxy)) * np.nan).reindex_like(interval_heights)

                if len(interval_df['Depositional Environment'].unique()) > 1:
                    print(f'error - multiple depositional environments in same interval in section {section}')
                    print(interval_df['Depositional Environment'].unique())
                    print(boundary_height, interval_boundary_heights[interval + 1])

                if len(interval_df['depositional age'].unique()) > 1:
                    print(f'error - multiple depositional ages in same interval in section {section}')
                    print(interval_df['depositional age'].unique())
                    print(boundary_height, interval_boundary_heights[interval + 1])

                intervals.append(interval)
                interval_dep_env = interval_df['Depositional Environment'].unique()[0]

                # TODO: put this into the dataframe after clustering
                interval_dep_age = interval_df['depositional age'].unique()[0]

                # here, check if we need to split the section at all, or if it already meets our criteria
                converged = check_convergence(interval_proxy, interval_heights, pvalue_thresh = pvalue_thresh)

                top_idx = len(interval_heights) - 1

                if converged:
                    clusters[interval][:] = section_cluster_number

                else:

                    while any(np.isnan(clusters[interval])):

                        group_not_found = True

                        lower_bound = np.where(np.isnan(clusters[interval]))[0][0]
                        upper_bound = top_idx

                        if len(np.where(np.isnan(clusters[interval]))[0]) > 1:
                            while group_not_found:

                                converged = check_convergence(interval_proxy.iloc[lower_bound:upper_bound + 1], interval_heights.iloc[lower_bound:upper_bound + 1], pvalue_thresh = pvalue_thresh)

                                # if group doesn't need to be split any more, assign to group
                                if converged:
                                    clusters[interval].iloc[lower_bound:upper_bound + 1] = int(section_cluster_number)
                                    group_not_found = False
                                    section_cluster_number += 1

                                else:
                                    upper_bound -= 1

                        # if only 1 sample left, assign to new cluster
                        else:
                            clusters[interval].iloc[np.where(np.isnan(clusters[interval]))[0][0]] = section_cluster_number
                            section_cluster_number += 1
                            group_not_found = False

                clusters[interval] = clusters[interval].astype(int)
                unique_clusters = np.unique(clusters[interval])

                c = 0
                for cluster in unique_clusters:

                    cluster_idx = np.where(clusters[interval] == cluster)[0]

                    # if only 1 data point, we'll just mark the sample as exclude = False instead of calculating cluster center
                    if len(cluster_idx) == 1:
                        no_downsample_idx.append(interval_df.index.tolist()[cluster_idx[0]])

                    # calculate cluster centers
                    else:
                        cluster_center_intervals.append(interval)

                        cluster_centers[interval].append((np.mean(interval_proxy[clusters[interval] == cluster]), np.mean(interval_heights[clusters[interval] == cluster])))
                        cluster_std[interval].append(np.std(interval_proxy[clusters[interval] == cluster]))
                        cluster_numbers.append(cluster)
                        # if not using cluster centers, find the index of the observation with the proxy value closest to the mean of each cluster
                        if nearest_point:
                            row_ix = np.asarray(np.where(clusters[interval] == cluster))[0]
                            if row_ix.shape[0] > 1:
                                # TODO: should we just use the sample w/ the proxy value closest to the center proxy value (disregarding the height)?
                                # calculate distance between cluster center and each data point in the cluster
                                current_center = cluster_centers[interval][c]
                                distance = np.zeros(row_ix.shape[0]) * np.nan
                                for i in np.arange(len(row_ix)):
                                    # this version calculates distance from cluster center using both heights and proxy values
                                    #distance[i] = np.linalg.norm(current_center - X[row_ix[i]])

                                    # this version only uses proxy values to calculate distance from cluster center
                                    distance[i] = np.abs(current_center[0] - interval_proxy.iloc[row_ix[i]])

                                closest_idx = row_ix[np.argmin(distance)]
                                closest_df_idx.append(interval_df.index.tolist()[closest_idx])

                            # if only 1 sample in cluster, keep it
                            else:
                                closest_df_idx.append(interval_df.index.tolist()[row_ix[0]])
                        c += 1

        # calculate cluster centers
        # if not nearest_point:
        for interval in intervals:
            if interval == intervals[0]:
                section_clusters = np.asarray(clusters[interval])
                if interval in cluster_center_intervals:
                    section_centers = np.asarray(cluster_centers[interval])
                    section_center_std = np.asarray(cluster_std[interval])

            else:
                section_clusters = np.concatenate([section_clusters, np.asarray(clusters[interval])])
                if interval in cluster_center_intervals:
                    section_centers = np.concatenate([section_centers, np.asarray(cluster_centers[interval])])
                    section_center_std =  np.concatenate([section_center_std, np.asarray(cluster_std[interval])])

        if nearest_point:
            # mark the samples we're keeping as Exclude? = False, and the rest as Exclude? = True
            # don't change samples that didn't have data for the downsampled proxy anyway
            sample_df_downsampled['Exclude?'][(sample_df_downsampled['section'] == section) & (~np.isnan(sample_df_downsampled[proxy]))] = True
            sample_df_downsampled['Exclude?'].loc[closest_df_idx] = False
            sample_df_downsampled['Exclude?'].loc[no_downsample_idx] = False
            sample_df_downsampled['cluster'].loc[section_df.index.tolist()] = section_clusters

        else:
            # mark all the samples in section as Exclude? = True, unless they don't have data for the downsampled proxy
            sample_df_downsampled['Exclude?'][(sample_df_downsampled['section'] == section) & (~np.isnan(sample_df_downsampled[proxy]))] = True
            sample_df_downsampled['Exclude?'].loc[no_downsample_idx] = False
            sample_df_downsampled['Exclude?'].loc[cluster_center_idx] = False # this only applies to clusters that only contain 1 data point
            sample_df_downsampled['cluster'].loc[section_df.index.tolist()] = section_clusters

            # if depositional age is present, save in dataframe (if not, set to nan)
            if str(interval_dep_age) != 'nan':
                dep_age_vec = [interval_dep_env] * section_centers.shape[0]
            else:
                dep_age_vec = [np.nan] * section_centers.shape[0]

            # add cluster centers to DataFrame
            downsampled_section_df = pd.DataFrame({'section': [section] * section_centers.shape[0],
                                               'height': section_centers[:, 1],
                                               proxy: section_centers[:, 0],
                                               proxy + '_std': section_center_std,
                                               'superposition?': [True] * section_centers.shape[0],
                                               'depositional age': dep_age_vec,
                                                'Exclude?': [False] * section_centers.shape[0],
                                                'Depositional Environment': [interval_dep_env] * section_centers.shape[0],
                                                  'cluster': cluster_numbers})



            sample_df_downsampled = pd.concat([sample_df_downsampled, downsampled_section_df], ignore_index = True)

    sample_df_downsampled.sort_values(by = ['section', 'height'], inplace = True)

    sample_df_downsampled['Exclude?'] = sample_df_downsampled['Exclude?'].astype(bool)

    sample_df_downsampled.reset_index(inplace = True, drop = True)

    return sample_df_downsampled

# helper function for checking whether a group of samples meets our criteria
# inputs = pandas series
def check_convergence(proxy, heights, pvalue_thresh = 0.05):
    lower_proxy, lower_heights, upper_proxy, upper_heights = split_data(proxy, heights)

    # if only 2-3 samples, split up if the standard deviation is > 0.5 (other criteria may not work well)
    if (len(lower_proxy) == 1) or (len(upper_proxy) == 1):
        if np.std(np.concatenate([lower_proxy, upper_proxy])) > 0.5:
            meets_criteria = False

        else:
            meets_criteria = True

    # if superposition between samples in the current segment is unknown, place all in the same group
    elif len(np.unique(np.concatenate([lower_heights, upper_heights]))) == 1:

        meets_criteria = True

    else:
        if (len(upper_proxy) > 2) and (len(lower_proxy) > 2):
            # check if samples from upper/lower halves of group are drawn from the same underlying distribution
            _, pvalue = ks_2samp(lower_proxy, upper_proxy, alternative = 'two-sided')

        else:
            if (np.abs(np.mean(upper_proxy) - np.mean(lower_proxy)) > 1):
                pvalue = 0
            else:
                pvalue = 1

        # check slopes of upper/lower halves
        if (len(lower_proxy) > 1) and (len(upper_proxy) > 1) and (len(np.unique(upper_heights) > 1)) and (len(np.unique(lower_heights) > 1)):
            # fit lines through the upper and lower groups of samples
            upper_p = np.polyfit(np.arange(len(upper_heights)), upper_proxy, deg = 1)
            lower_p = np.polyfit(np.arange(len(lower_heights)), lower_proxy, deg = 1)

            upper_p_sign = np.sign(upper_p[0])
            lower_p_sign = np.sign(lower_p[0])

            # if signs don't match (and it isn't basically a flat line), need to split again --> manually set the p-value to 0
            # and (np.abs(upper_p[0]) > 0.2) and (np.abs(lower_p[0]) > 0.2):
            if (upper_p_sign != lower_p_sign) or (np.abs(np.abs(upper_p[0]) - np.abs(lower_p[0])) > 0.2):
                pvalue = 0

        # calculate slope with x = sample number (instead of height) -- less sensitive to total section thickness
        if len(np.concatenate([lower_heights, upper_heights])) > 2:
            total_p = np.polyfit(np.arange(len(lower_heights) + len(upper_heights)), np.concatenate([lower_proxy, upper_proxy]), deg = 1)
            residual_autocorr = pd.Series(np.concatenate([lower_proxy, upper_proxy]) -  (np.arange(len(lower_heights) + len(upper_heights)) * total_p[0] + total_p[1])).autocorr()

            if residual_autocorr > 0.5:
                pvalue = 0

            # if total number of samples in group is <=3, tests may not work well -- split if the standard deviation is >0.5

        if (pvalue < pvalue_thresh):# or (pvalue_means < pvalue_thresh):
            meets_criteria = False

        else:
            meets_criteria = True

    return meets_criteria

# helper function for splitting group of samples into lower/upper halves based on height
# works on pandas series
def split_data(proxy, heights):

    sort_idx = np.argsort(heights.values)
    lower_idx = sort_idx[0:int(np.ceil(len(heights)/2))]
    upper_idx = sort_idx[int(np.ceil(len(heights)/2)):]

    lower_heights = heights.iloc[lower_idx]
    lower_proxy = proxy.iloc[lower_idx]

    upper_heights = heights.iloc[upper_idx]
    upper_proxy = proxy.iloc[upper_idx]

    # keep track of where to put cluster assignments using pandas series indices
    return lower_proxy, lower_heights, upper_proxy, upper_heights
