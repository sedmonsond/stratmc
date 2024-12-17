import pickle
import sys
import warnings

import arviz as az
import numpy as np
import pandas as pd
from scipy.fftpack import fft, fftfreq
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", ".*The group X_new is not defined in the InferenceData scheme.*")
warnings.filterwarnings("ignore", ".*X_new group is not defined in the InferenceData scheme.*")

def load_data(sample_file, ages_file, proxies = ['d13c'], proxy_sigma_default = 0.2, drop_excluded_samples = True, drop_excluded_ages = True, combine_no_superposition = False):
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
        Whether to remove samples with ``Exclude? = True`` from the ``sample_df``; defaults to ``True``. If excluded samples are not dropped, their ages will be passively tracked within the inference model (but they will not be considered during the proxy signal reconstruction).

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
    sample_df = combine_duplicates(sample_df, proxies, proxy_sigma_default, combine_no_superposition = combine_no_superposition)

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

def combine_duplicates(sample_df, proxies, proxy_sigma_default = 0.1, combine_no_superposition = False):
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

    if combine_no_superposition:
        sample_df = sample_df[(~sample_df['Exclude?'].values.astype(bool))]
    else:
        no_superposition_sample_df = sample_df[~sample_df['superposition?']]
        no_superposition_sample_df.reset_index(inplace = True, drop = True)
        sample_df = sample_df[(~sample_df['Exclude?'].values.astype(bool)) & (sample_df['superposition?'].values.astype(bool))]

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

            # remove the duplicate samples from sample_df
            sample_df.drop(index = duplicate_sub_idx, inplace = True)

            duplicate_dicts.append(duplicate_dict)

    # add combined data to dataframe
    for duplicate in duplicate_dicts:
        sample_df = pd.concat([sample_df, pd.DataFrame.from_dict(duplicate)], ignore_index = True)

    # put the excluded samples back
    if excluded_sample_df.shape[0] > 0:
        sample_df = pd.concat([sample_df, excluded_sample_df], ignore_index = True)

    # put samples w/out superposition information back
    if (not combine_no_superposition):
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
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`build_model() <stratmc.model.build_model>` in :py:mod:`stratmc.model` (the output of :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`).

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

    Note that if ``method = 'all'``, rate is returned in mm/year, and duration is returned in years. If ``method = 'successive'``, rate is returned in m/Myr, and duration is returned in Myr. Input data are assumed to have units of meters and millions of years. Used as input to :py:meth:`sadler_plot() <stratmc.plotting.sadler_plot>` and :py:meth:`accumulation_rate_stratigraphy() <stratmc.plotting.accumulation_rate_stratigraphy>` in :py:mod:`stratmc.plotting`.

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
    Extend age models calculated using downsampled proxy observations from :py:meth:`downsample() <bayestrat.data.downsample>` to the full set of proxy observations.

    .. todo::
        Remove? Shouldn't be necessary since ages for excluded samples can now be tracked w/in the model
    .. todo::
        Check behavior with excluded samples


    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`build_model() <bayestrat.model.build_model>` in :py:mod:`bayestrat.model`.

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

def downsample(sample_df, ages_df, N = 5000, corr_coef_min = 0.90, proxy = 'd13c', mode = 'clusters', keep = 'best', keep_seed = None, flexible_cluster_size = True, check_cluster_std_dev = False, check_cluster_residual_autocorr = False, check_cluster_autocorr = False, cluster_std_max = 0.5, check_residuals = False, max_interp_residual = 1, resample_with_lowest_n = True, compare_white_noise_fft = False, relative_structure_min = 0.90, N_white_noise = 1000, best_criteria = 'corr_coef', **kwargs):
    """
    Downsample a set of proxy observations. Computes the correlation coefficient between N downsampled (by randomly grouping or subsampling the data, depending on ``mode``) versions of the data and the original data set. Returns the solution that meets the correlation coefficient minimum with the lowest number of downsampled data points. See input parameter descriptions for additional details.

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    N: int
        Number of random sample groupings to test. Defaults to 5,000.

    mode: str
        Method for downsampling data. The 'clusters` mode splits samples into groups by testing ``N`` random sample groupings. Each group then is represented by its centroid, and the downsampled data set is comprised of these centroids (with uncertainty equal to population standard deviation). The 'data` mode instead retains a subset of the original data points, and discards the others (by marking them as ``Exclude? = True`` in the ``downsampled_data`` DataFrame).

    keep: str
        If there are multiple solutions that satisfy ``corr_coef_min`` using the minimum possible number of data points, whether to return the best one of these solutions ('best`), or a random solution ('random`). Defaults to 'best`.

    flexible_cluster_size: bool
        Whether to consider solutions with 1 more number of clusters/data points than the minimum when ``keep`` is 'random`. Defaults to ``True``.

    proxy: str, optional
        Proxy to downsample. Defaults to 'd13c`.

    sections: list(str) or numpy.array(str), optional
        List of sections to downsample. Defaults to all sections in ``sample_df``.

    corr_coef_min: float or dict{float}, optional
        Minimum acceptable correlation coefficient. For each section, the algorithm selects the smallest downsampled data set that meets this threshold. If multiple solutions with this minimum number of data points exist, then the solution with the highest correlation coefficient is selected if ``keep`` is 'best`, while a random one of these solutions is selected if ``keep`` is 'random`. Must be in ``[0, 1]``; defaults to 0.9. Pass as a dictionary to specify a different value for each section.

    Returns
    -------
    downsampled_data: pandas.DataFrame
        :class:`pandas.DataFrame` containing downsampled proxy data. All samples are still included in the DataFrame, but samples that were excluded during downsampling are marked ``Exclude? = True``.

    solution_corr_coefs: dict
        Dictionary with the correlation coefficients for chosen solutions; keys are section names.

    """

    sample_df_downsampled = sample_df.copy()
    sample_df_downsampled['cluster'] = np.nan

    if 'sections' in kwargs:
            sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df_downsampled['section'])

    if type(corr_coef_min) != dict:
        temp = corr_coef_min
        corr_coef_min = {}
        for section in sections:
            corr_coef_min[section] = temp

    downsampled_df = {}
    solution_corr_coefs = {}

    downsampled_df[proxy] = pd.DataFrame(columns = ['section', 'height', proxy, proxy + '_std', 'superposition?', 'Exclude?', 'Depositional Environment'])

    for section in tqdm(sections):
        print(f'Downsampling {section}')

        section_df = sample_df[(sample_df['section']==section) & (~sample_df['Exclude?'].astype(bool))].dropna(subset = proxy)

        heights = section_df['height'].values
        proxy_vec = section_df[proxy].values

        if len(heights) > 2:
            # grab required boundaries (changes in depositional environment, superposition, depositional ages)
            required_boundaries = get_boundaries(sample_df, ages_df, proxy, section)

            # grab list of non-required candidate boundaries (heights between samples that aren't already in required_bondaries_
            candidate_boundaries = get_candidate_boundaries(heights, required_boundaries)

            # make evenly height grid for fft
            height_grid = get_height_grid(heights)

            # interpolate proxy values to height grid
            proxy_grid = get_proxy_grid(proxy_vec, heights, height_grid)

            max_n_clusters = len(heights)

            # not counting upper/lower bounds of section, + 1 because number of clusters is 1 more than # of boundaries
            min_n_clusters = len(required_boundaries) - 2 + 1

            max_n_data = len(heights)

            # if we're going to compare with white noise, go ahead and run fft for section
            if compare_white_noise_fft:

                _, full_powers = get_powers(proxy_grid, height_grid)

                data_mean = np.mean(proxy_grid)
                data_std = np.std(proxy_grid)

                power_diff_white = np.ones(N_white_noise) * np.nan

                for i in tqdm(np.arange(N_white_noise)):
                    rng = np.random.default_rng(seed = i)
                    white_noise = rng.normal(data_mean, data_std, len(height_grid))
                    _, white_noise_powers = get_powers(white_noise, height_grid)
                    power_diff_white[i] = np.mean(np.abs(white_noise_powers - full_powers))

                # mean difference to use in comparisons
                mean_power_diff_white = np.mean(power_diff_white)

            centroids = {}
            centroid_std = {}
            n_clusters = {}
            cluster_bounds = {}
            random_data_idx = {}
            corr_coef = np.ones(N) * np.nan
            n_clusters = np.ones(N) * np.nan
            n_data_points = np.ones(N) * np.nan

            if compare_white_noise_fft:
                power_diff_means = np.ones(N) * np.nan

            if check_residuals or best_criteria == 'residuals':
                mean_residuals =  np.ones(N) * np.nan

            if check_cluster_std_dev or check_cluster_residual_autocorr or check_cluster_autocorr or check_residuals or compare_white_noise_fft:
                check2 = []

            for i in tqdm(np.arange(N)):
                rng = np.random.default_rng(seed = i)

                if mode == 'clusters':
                    n_clusters[i] = rng.choice(np.arange(min_n_clusters, max_n_clusters + 1), 1)[0]
                    cluster_bounds[i] = required_boundaries.copy()

                    # upper and lower bounds don't split the section (+2)
                    # number of boundaries is 1 less than number of clusters (-1)
                    n_new_bounds = n_clusters[i] - len(cluster_bounds[i]) + 2 - 1

                    new_bounds = rng.choice(candidate_boundaries, size = int(n_new_bounds), replace = False)

                    cluster_bounds[i] = np.sort(np.concatenate([cluster_bounds[i], new_bounds]))

                    cluster_bounds[i] = remove_extra_bounds(heights, cluster_bounds[i])

                    centroids[i], centroid_std[i] = get_centroids(proxy_vec, heights, cluster_bounds[i])

                    if check_cluster_std_dev:
                        check2.append(all(centroid_std[i] <= cluster_std_max))

                    elif check_cluster_residual_autocorr:
                        # residual_autocorr = get_cluster_autocorr(proxy_vec, heights, cluster_bounds[i])
                        # check2.append(all(np.array(residual_autocorr) <= 0.5))
                        residual_autocorr, shuffled_autocorr = get_cluster_autocorr_vs_shuffled(proxy_vec, heights, cluster_bounds[i], mode = 'residual')
                        check2.append(all(np.array(residual_autocorr) <= (np.array(shuffled_autocorr))))

                    elif check_cluster_autocorr:
                        # residual_autocorr = get_cluster_autocorr(proxy_vec, heights, cluster_bounds[i])
                        # check2.append(all(np.array(residual_autocorr) <= 0.5))
                        residual_autocorr, shuffled_autocorr = get_cluster_autocorr_vs_shuffled(proxy_vec, heights, cluster_bounds[i], mode = 'data')
                        check2.append(all(np.array(residual_autocorr) <= (np.array(shuffled_autocorr))))

                    interp_proxy = np.interp(height_grid, centroids[i][:, 1], centroids[i][:, 0])

                    if compare_white_noise_fft:
                        _, interp_powers = get_powers(interp_proxy, height_grid)

                        power_diff_means[i] = np.mean(np.abs(interp_powers - full_powers))


                elif mode == 'data':
                    random_data_idx[i] = []
                    n_data_points[i] = rng.choice(np.arange(np.max([min_n_clusters, 2]), max_n_data + 1), 1)[0]

                    candidate_idx = section_df.index.tolist()

                    # first, select 1 data point each from w/in each 'required' interval -- this ensures that all age constraints and depositional environments are represented
                    for interval, boundary_height in enumerate(required_boundaries[:-1]):
                        above = heights >= boundary_height
                        below = heights < required_boundaries[interval + 1]

                        cluster_idx = np.array(section_df.index.tolist())[above & below]

                        random_data_idx[i].append(rng.choice(cluster_idx, 1)[0])

                        # remove the chosen sample from the list of candidate data points
                        candidate_idx.remove(random_data_idx[i][-1])

                    # calculate how many additional data points we need to reach target number
                    n_remaining = n_data_points[i] - len(random_data_idx[i])

                    random_data_idx[i] += list(rng.choice(candidate_idx, int(n_remaining), replace = False))

                    random_data_idx[i].sort()

                    interp_proxy = np.interp(height_grid, section_df['height'].loc[random_data_idx[i]], section_df[proxy].loc[random_data_idx[i]])

                    if check_residuals or best_criteria == 'residuals':
                        if len(random_data_idx[i]) == len(heights):
                            if check_residuals:
                                check2.append(True)

                            mean_residuals[i] = 0

                        else:
                            for idx in random_data_idx[i]:
                                if idx in candidate_idx:
                                    candidate_idx.remove(idx)

                                candidate_idx.sort()

                            # interpolate downsampled signal to heights of samples that weren't chosen
                            proxy_interp_to_excluded_data = np.interp(section_df['height'].loc[candidate_idx], section_df['height'].loc[random_data_idx[i]], section_df[proxy].loc[random_data_idx[i]])

                            proxy_interp_residuals = np.abs(proxy_interp_to_excluded_data - section_df[proxy].loc[candidate_idx])

                            mean_residuals[i] = np.mean(proxy_interp_residuals)

                            if check_residuals:
                                check2.append(all(proxy_interp_residuals <= max_interp_residual))

                    if compare_white_noise_fft:
                        _, interp_powers = get_powers(interp_proxy, height_grid)

                        power_diff_means[i] = np.mean(np.abs(interp_powers - full_powers))

                corr_coef[i] = np.corrcoef(proxy_grid, interp_proxy)[0, 1]

            # full list of configurations that meet the criteria
            if check_cluster_std_dev or check_cluster_residual_autocorr or check_cluster_autocorr or check_residuals:
                check1 = (corr_coef >= corr_coef_min[section])
                above = check1 & check2

            elif compare_white_noise_fft:
                check1 = (corr_coef >= corr_coef_min[section])
                relative_structure_retained = 1 - (power_diff_means/mean_power_diff_white)
                check2 = relative_structure_retained >= relative_structure_min
                above = check1 & check2

            else:
                above = corr_coef >= corr_coef_min[section]

            keep_idx_list = np.argwhere(above)

            if mode == 'clusters':
                # minimum number of clusters in list
                min_n_clusters = np.min(n_clusters[keep_idx_list])

                if (keep == 'random') and (flexible_cluster_size):
                    n_cluster_idx = np.where((n_clusters[keep_idx_list] == min_n_clusters) | (n_clusters[keep_idx_list] == min_n_clusters + 1))[0]
                else:
                    # get indices (within candidate list) where number of clusters is equal to the minimum
                    n_cluster_idx = np.where(n_clusters[keep_idx_list] == min_n_clusters)[0]

                if (keep == 'best') or (len(n_cluster_idx) == 1):
                    best_idx_temp = np.argmax(corr_coef[keep_idx_list][n_cluster_idx])
                    best_idx = keep_idx_list[n_cluster_idx[best_idx_temp]][0]

                elif (keep == 'random') and (len(n_cluster_idx) > 1):
                    keep_rng = np.random.default_rng(seed = keep_seed)
                    best_idx_temp = keep_rng.choice(n_cluster_idx, 1)[0]
                    best_idx = keep_idx_list[best_idx_temp][0]

                # once we've found the best configuration, gather info for dataframe
                keep_centroids = centroids[best_idx]
                keep_centroid_std = centroid_std[best_idx]

                # assign clusters to groups
                cluster_center_idx = []

                dep_env_list = []
                dep_age_list = []

                keep_centroid_idx = list(np.arange(keep_centroids.shape[0]))

                for interval, boundary_height in enumerate(cluster_bounds[best_idx][:-1]):
                    #plt.axhline(h, color = 'indianred', linestyle = 'dashed', zorder = 0)
                    above = heights >= boundary_height
                    below = heights < cluster_bounds[best_idx][interval + 1]

                    cluster_idx = np.array(section_df.index.tolist())[above & below]
                    sample_df_downsampled['cluster'].loc[cluster_idx] = interval

                    # save indices of clusters with only n = 1 data point
                    if len(heights[above & below]) == 1:
                        row_idx = np.where(heights == heights[above & below])[0]
                        cluster_center_idx += list(np.array(section_df.index.tolist())[row_idx])
                        keep_centroid_idx.remove(interval)

                    # if more than one sample, grab depositional environment and depositional age info for adding centers to dataframe
                    else:
                        interval_dep_env_list = np.unique(section_df['Depositional Environment'].iloc[above & below])

                        if len(interval_dep_env_list) > 1:
                            print('multiple depositional environments in same group')
                        else:
                            dep_env_list.append(interval_dep_env_list[0])

                        interval_dep_age_list = np.unique(section_df['depositional age'].iloc[above & below].astype(str))

                        if len(interval_dep_age_list) > 1:
                            print('multiple depositional ages in same group')

                        else:
                            if interval_dep_age_list[0] != 'nan':
                                dep_age_list.append(interval_dep_age_list[0])
                            else:
                                dep_age_list.append(np.nan)


            elif mode == 'data':
                # minimum number of clusters in list
                min_n_data = int(np.min(n_data_points[keep_idx_list]))

                # NOTE -- this can rarely cause problems if it doesn't re-find the solution from the previous part w/ the sufficiently high correlation coeficient. in this case, probably try re-running w/ higher N
                if resample_with_lowest_n:
                    if check_residuals:
                        check2 = []
                    print(f'Resampling {section}')
                    for i in tqdm(np.arange(N)):

                        rng = np.random.default_rng(seed = i)

                        random_data_idx[i] = []

                        if flexible_cluster_size:
                            if min_n_data < len(heights):
                                n_data_points[i] = rng.choice([min_n_data, min_n_data + 1], 1)
                            else:
                                n_data_points[i] = min_n_data
                        else:
                            n_data_points[i] = min_n_data

                        candidate_idx = section_df.index.tolist()

                        # first, select 1 data point each from w/in each 'required' interval -- this ensures that all age constraints and depositional environments are represented
                        for interval, boundary_height in enumerate(required_boundaries[:-1]):
                            above = heights >= boundary_height
                            below = heights < required_boundaries[interval + 1]

                            cluster_idx = np.array(section_df.index.tolist())[above & below]

                            random_data_idx[i].append(rng.choice(cluster_idx, 1)[0])

                            # remove the chosen sample from the list of candidate data points
                            candidate_idx.remove(random_data_idx[i][-1])

                        # calculate how many additional data points we need to reach target number
                        n_remaining = n_data_points[i] - len(random_data_idx[i])

                        random_data_idx[i] += list(rng.choice(candidate_idx, int(n_remaining), replace = False))

                        random_data_idx[i].sort()

                        interp_proxy = np.interp(height_grid, section_df['height'].loc[random_data_idx[i]], section_df[proxy].loc[random_data_idx[i]])

                        corr_coef[i] = np.corrcoef(proxy_grid, interp_proxy)[0, 1]

                        if compare_white_noise_fft:
                            _, interp_powers = get_powers(interp_proxy, height_grid)

                            power_diff_means[i] = np.mean(np.abs(interp_powers - full_powers))

                        if check_residuals or best_criteria == 'residuals':

                            if len(random_data_idx[i]) == len(heights):
                                if check_residuals:
                                    check2.append(True)

                                mean_residuals[i] = 0

                            else:
                                for idx in random_data_idx[i]:
                                    if idx in candidate_idx:
                                        candidate_idx.remove(idx)

                                    candidate_idx.sort()

                                # interpolate downsampled signal to heights of samples that weren't chosen
                                proxy_interp_to_excluded_data = np.interp(section_df['height'].loc[candidate_idx], section_df['height'].loc[random_data_idx[i]], section_df[proxy].loc[random_data_idx[i]])

                                proxy_interp_residuals = np.abs(proxy_interp_to_excluded_data - section_df[proxy].loc[candidate_idx])
                                mean_residuals[i] = np.mean(proxy_interp_residuals)
                                if check_residuals:
                                    check2.append(all(proxy_interp_residuals <= max_interp_residual))

                    if compare_white_noise_fft:
                        check1 = corr_coef >= corr_coef_min[section]
                        relative_structure_retained = 1 - (power_diff_means/mean_power_diff_white)
                        check2 = relative_structure_retained >= relative_structure_min
                        above = check1 & check2

                    if check_residuals:
                        check1 = corr_coef >= corr_coef_min[section]
                        above = check1 & check2

                    else:
                        above = corr_coef >= corr_coef_min[section]

                    # solutions that meet minimum criteria
                    keep_idx_list = np.argwhere(above)

                    if len(keep_idx_list) == 0:
                        print('Try running with higher N; no viable solutions found after resampling')

                # NOTE: changed so size can be flexible (+1 larger than the minimum) even if we're looking for the best solution, not just a random one
                if flexible_cluster_size:  # (keep == 'random') and
                    n_data_idx = np.where((n_data_points[keep_idx_list] == min_n_data) | (n_data_points[keep_idx_list] == min_n_data + 1))[0]

                else:
                    # get indices (within candidate list) where number of clusters is equal to the minimum
                    n_data_idx = np.where(n_data_points[keep_idx_list] == min_n_data)[0]

                # TODO: use different criteria to find best solution
                if (keep == 'best') or (len(n_data_idx) == 1):
                    # out of the viable solutions, choose the one with the highest correlation coefficient
                    if best_criteria == 'corr_coef':
                        best_idx_temp = np.argmax(corr_coef[keep_idx_list][n_data_idx])

                    # out of the viable solutions, choose the one with the lowest (mean) residuals between the interpolated signal and the excluded data points
                    elif best_criteria == 'residuals':
                        best_idx_temp = np.argmin(mean_residuals[keep_idx_list][n_data_idx])

                    elif best_criteria == 'structure_retained':
                        best_idx_temp = np.argmax(relative_structure_retained[keep_idx_list][n_data_idx])

                    best_idx = keep_idx_list[n_data_idx[best_idx_temp]][0]


                elif (keep == 'random') and (len(n_data_idx) > 1):
                    keep_rng = np.random.default_rng(seed = keep_seed)

                    best_idx_temp = keep_rng.choice(n_data_idx, 1)[0]

                    best_idx = keep_idx_list[best_idx_temp][0]

            solution_corr_coefs[section] = corr_coef[best_idx]

            if mode == 'data':
                # mark the samples we're keeping as Exclude? = False, and the rest as Exclude? = True
                # don't change samples that didn't have data for the downsampled proxy anyway
                sample_df_downsampled.loc[(sample_df_downsampled['section'] == section) & (~np.isnan(sample_df_downsampled[proxy])), 'Exclude?'] = True
                sample_df_downsampled.loc[random_data_idx[best_idx], 'Exclude?'] = False

            elif mode == 'clusters':
                # mark all the samples in section as Exclude? = True, unless they don't have data for the downsampled proxy
                sample_df_downsampled.loc[(sample_df_downsampled['section'] == section) & (~np.isnan(sample_df_downsampled[proxy])), 'Exclude?'] = True
                sample_df_downsampled.loc[cluster_center_idx, 'Exclude?'] = False # this only applies to clusters that only contain 1 data point

                # add cluster centers to DataFrame
                downsampled_section_df = pd.DataFrame({'section': [section] * len(keep_centroids[keep_centroid_idx, 0]),
                                                   'height': keep_centroids[keep_centroid_idx, 1],
                                                   proxy: keep_centroids[keep_centroid_idx, 0],
                                                   proxy + '_std': keep_centroid_std[keep_centroid_idx],
                                                   'superposition?': [True] * len(keep_centroids[keep_centroid_idx, 0]),
                                                   'depositional age': dep_age_list,
                                                    'Exclude?': [False] * len(keep_centroids[keep_centroid_idx, 0]),
                                                    'Depositional Environment': dep_env_list,
                                                     'cluster': keep_centroid_idx})

                sample_df_downsampled = pd.concat([sample_df_downsampled, downsampled_section_df], ignore_index = True)

    sample_df_downsampled['Exclude?'] = sample_df_downsampled['Exclude?'].astype(bool)

    return sample_df_downsampled, solution_corr_coefs

def get_cluster_autocorr(proxy, heights, boundaries, mode = 'residual'):
    autocorr = []
    for interval, boundary_height in enumerate(boundaries[:-1]):
        above = heights >= boundary_height
        below = heights < boundaries[interval + 1]

        proxy_cluster = proxy[above & below]

        # can't compute without at least 3 data points
        if len(proxy_cluster) > 2:
            if mode == 'residual':
                p = np.polyfit(np.arange(len(proxy_cluster)), proxy_cluster, deg = 1)
                autocorr_temp = pd.Series(proxy_cluster -  (np.arange(len(proxy_cluster)) * p[0] + p[1])).autocorr()

            elif mode == 'data':
                autocorr_temp = pd.Series(proxy_cluster).autocorr()
            autocorr.append(autocorr_temp)

    return autocorr

def get_cluster_autocorr_vs_shuffled(proxy, heights, boundaries, mode = 'residual'):
    autocorr = []
    shuffled_autocorr = []
    for interval, boundary_height in enumerate(boundaries[:-1]):
        above = heights >= boundary_height
        below = heights < boundaries[interval + 1]

        proxy_cluster = proxy[above & below]

        # can't compute without at least 3 data points
        if len(proxy_cluster) > 2:

            if mode == 'residual':
                p = np.polyfit(np.arange(len(proxy_cluster)), proxy_cluster, deg = 1)
                autocorr_temp = pd.Series(proxy_cluster -  (np.arange(len(proxy_cluster)) * p[0] + p[1])).autocorr()

            elif mode == 'data':
                autocorr_temp = pd.Series(proxy_cluster).autocorr()

            autocorr.append(autocorr_temp)

            shuffled_autocorr_temp = []
            for i in np.arange(10):
                rng = np.random.default_rng(seed = i)
                proxy_new = rng.permutation(proxy_cluster)

                if mode == 'residual':
                    p = np.polyfit(np.arange(len(proxy_new)), proxy_new, deg = 1)
                    autocorr_temp = pd.Series(proxy_new -  (np.arange(len(proxy_new)) * p[0] + p[1])).autocorr()

                elif mode == 'data':
                    autocorr_temp = pd.Series(proxy_new).autocorr()

                shuffled_autocorr_temp.append(autocorr_temp)

            shuffled_autocorr.append(np.mean(shuffled_autocorr_temp))


    return autocorr, shuffled_autocorr


def get_boundaries(sample_df, ages_df, proxy, section, environment = True, depositional_ages = True, superposition = True):
    """
    Helper function for :py:meth:`downsample() <stratmc.data.downsample>`. Returns list of height boundaries where the target section must be split into different groups. By default, inserts breaks between samples from different depositional environments, around groups of samples with the same depositional age, and around groups of samples without superposition information.

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing all proxy data.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints from all sections.

    proxy: str
        Name of proxy to be downsampled.

    section: str
        Name of target section.

    environment: bool
        Whether to insert breaks between different depositional environments.

    depositional_ages: bool
        Whether to insert breaks around groups of samples with the same depositional age constraint.

    superposition:
        Whether to insert breaks around groups of samples without superposition information.

    Returns
    -------
    boundary_heights: numpy.array
        Array containing required cluster boundaries.

    """

    section_df = sample_df[(sample_df['section']==section) & (~sample_df['Exclude?'].astype(bool))].dropna(subset = proxy)
    section_ages_df = ages_df[(ages_df['section']==section)  & (~ages_df['depositional?'])]

    heights = section_df['height'].values
    age_heights = section_ages_df['height'].values

    # create a list of interval boundary heights: 1) age constraint, 2) change in 'superposition?' boolean, 3) change in depositional environment
    boundary_heights = list(age_heights)

    section_unique_dep_env = section_df['Depositional Environment'].unique()
    section_dep_env = section_df['Depositional Environment'].values
    section_superposition = section_df['superposition?'].values
    section_dep_ages = section_df['depositional age'].values
    section_unique_dep_ages = list(section_df['depositional age'].astype(str).unique())

    if 'nan' in section_unique_dep_ages:
        section_unique_dep_ages.remove('nan')

    if superposition:
        # if there's a change in superposition boolean within the interval, need to split into a separate interval (w/ same lower and upper bound -- just add the height twice)
        if not (all(section_superposition)) or (all(~section_superposition)):
            # grab heights of samples without superposition
            super_heights = np.unique(section_df[~section_df['superposition?']]['height'])

            for h in super_heights:
                if h not in boundary_heights:
                    # append the height
                    boundary_heights += [h]
                    top_h_idx = np.where(section_df['height'] == h)[0][-1]

                # bound with height of overlying sample, if not already done or at top of section
                if (h != np.max(heights)):
                    if heights[top_h_idx + 1] not in super_heights:
                        boundary_heights.append(heights[top_h_idx + 1])

    if depositional_ages:
        # add boundaries around groups of samples with the same depositional age
        for dep_age in section_unique_dep_ages:
            # print(f'splitting depositional ages for section {section}')
            dep_age_idx = np.where(section_dep_ages == dep_age)[0]

            # assuming all the ages are in 1 chunk, add the base and the overlying sample
            if all(np.diff(dep_age_idx) == 1):

                # only add base if chunk isn't at base of section
                if dep_age_idx[0] != 0:
                    boundary_heights.append(heights[dep_age_idx[0]])

                # only add overlying sample if not at top of section (already bounded by another age constraint)
                if dep_age_idx[-1] != len(heights) - 1:
                    boundary_heights.append(heights[dep_age_idx[-1] + 1])

            else:
                print(f'samples with depositional age {dep_age} in section {section} are not in a continuous chunk - check that depositional age assignment is correct')
                if (len(dep_age_idx) > 1):
                        switch_idx = np.where(np.diff(dep_age_idx) != 1)[0]

                        boundary_heights += list(heights[dep_age_idx[switch_idx] + 1])

                        # add boundary above the uppermost chunk, unless it's the top of the secion  (in which case there should already be an age constraint)
                        if dep_age_idx[-1] != len(heights) - 1:
                            boundary_heights += list([heights[dep_age_idx[-1] + 1]])

    # add boundaries between different depositional environments
    if environment:
        if len(section_unique_dep_env) > 1:
            for env in section_unique_dep_env:
                env_idx = np.where(section_dep_env == env)[0]

                # add base of lowermost group, unless we're at the bottom of the section
                if env_idx[0] != 0:
                    boundary_heights.append(heights[env_idx[0]])

                # if all samples from this environment are in 1 chunk, just add the top boundary
                if (len(env_idx)) >= 1 and (all(np.diff(env_idx) == 1)):
                    # don't add to list if chunk is at top of the section (already bounded by an age constraint)
                    if (env_idx[-1] != len(heights) - 1):
                        boundary_heights.append(heights[env_idx[-1] + 1])

                else:
                    # if there's more than one sample from this environment (scenario with only 1 is covered above)
                    if (len(env_idx) > 1):
                        switch_idx = np.where(np.diff(env_idx) != 1)[0]

                        boundary_heights += list(heights[env_idx[switch_idx] + 1])

                        # add boundary above the uppermost chunk, unless it's the top of the secion  (in which case there should already be an age constraint)
                        if env_idx[-1] != len(heights) - 1:
                            boundary_heights += list([heights[env_idx[-1] + 1]])

    # sort interval boundaries, and get rid of any duplicate boundaries
    boundary_heights = np.sort(np.unique(boundary_heights))

    boundary_heights = remove_extra_bounds(heights, boundary_heights)


    return boundary_heights

def get_centroids(proxy, heights, bounds):
    """
    Helper function for :py:meth:`downsample() <stratmc.data.downsample>`; calculate the centroid for a group of samples.

    Parameters
    ----------
    proxy: numpy.array
        array containing proxy values for samples in group

    height: pandas.DataFrame
        array containing heights for samples in group

    bounds: np.array
        array containing heights of boundaries between groups

    Returns
    -------
    centroid: np.array
        Array containing centroid coordinates: [proxy_center, height_center]

    """

    centroids = np.ones((len(bounds) - 2 + 1, 2)) * np.nan
    centroid_std = np.ones(len(bounds) - 2 + 1) * np.nan
    for interval, boundary_height in enumerate(bounds[:-1]):

        above = heights >= boundary_height
        below = heights < bounds[interval + 1]

        centroids[interval, 0] = np.mean(proxy[above & below])
        centroids[interval, 1] = np.mean(heights[above & below])

        centroid_std[interval] = np.std(proxy[above & below])

        if len(heights[above & below]) == 0:
            print('no samples in interval, fix boundaries')


    return centroids, centroid_std

def remove_extra_bounds(heights, boundaries):
    """
    Helper function for :py:meth:`downsample() <stratmc.data.downsample>`; removes duplicate or extraneous boundaries from list of candidate cluster boundaries.

    Parameters
    ----------
    proxy: numpy.array
        array containing proxy values for samples in group

    height: pandas.DataFrame
        array containing heights for samples in group

    bounds: np.array
        array containing heights of boundaries between groups

    Returns
    -------
    centroid: np.array
        Array containing centroid coordinates: [proxy_center, height_center]

    """

    # check that there are samples between all boundaries. if not, get rid of the upper boundary

    boundaries = list(boundaries)

    finished = False
    while not finished:
        for interval, height in enumerate(boundaries[:-1]):
            above = heights >= height
            below = heights < boundaries[interval + 1]

            if len(heights[above & below]) == 0:
                # remove the current boundary if there aren't any samples in interval
                # note -- if we remove the upper boundary, may end up w/ >1 sample in group b/c the upper boundary is non-inclusive
                boundaries.remove(boundaries[interval])
                break

            if interval == len(boundaries) - 2:
                finished = True

    return np.array(boundaries)

def get_powers(proxy, height):
    # proxy_grid_detrend = detrend(proxy_grid)

    N = len(proxy)
    dh = np.diff(height)[0]

    yf = fft(proxy)

    # xf, powers = power_freq(yf, N, dh)

    freq = fftfreq(N, dh)[:N//2]
    powers = 2.0/N * np.abs(yf[1:N//2])

    return freq, powers


def get_height_grid(heights):
    """
    Helper function for :py:meth:`downsample() <stratmc.data.downsample>`; returns an evenly spaced grid of heights spanning the same range as the input heights. Spacing is equal to half of the minimum height between samples.

    Parameters
    ----------
    heights: numpy.array
        Array containing sample heights for a single section.

    Returns
    -------
    height_grid: np.array
        Array of evenly spaced heights spanning the range in ``heights``.

    """

    dy = np.diff(heights)
    dy = dy[(dy > 0)]

    # spacing = minimum distance between samples
    # could also divide this by 2, just more computationally expensive
    min_dy = np.min(dy)/2

    height_grid = np.arange(np.min(heights), np.max(heights) + min_dy, min_dy).ravel()

    return height_grid


def get_proxy_grid(proxy, height, height_grid):
    """
    Helper function for :py:meth:`downsample() <stratmc.data.downsample>`. Interpolates proxy values for a given section to the new heights specified in ``height_grid`` (from :py:meth:`get_height_grid() <stratmc.data.get_height_grid>`.)

    Parameters
    ----------
    proxy: numpy.array
        Array containing proxy values corresponding to the heights in ``heights``.

    heights: numpy.array
        Array containing sample heights for a single section.

    height_grid: numpy.array
        Array containing heights at which to interpolate the proxy values.

    Returns
    -------
    proxy_grid: np.array
        Array of interpolated proxy values corresponding to the heights in ``height_grid``.

    """

    proxy_grid = np.interp(height_grid, height, proxy)

    return proxy_grid


def get_candidate_boundaries(heights, required_boundary_heights):
    """
    Helper function for :py:meth:`downsample() <stratmc.data.downsample>`; returns list of potential cluster boundaries.

    Parameters
    ----------
    heights: numpy.array
        Array containing sample heights for a single section.

    required_boundary_heights: numpy.array
        Array containing heights of required boundaries (from :py:meth:`get_boundaries() <bayestrat.data.get_boundaries>`)

    Returns
    -------
    candidate_boundaries: np.array
        Array of potential cluster boundaries (excluding required boundaries from ``required_boundary_heights``).

    """

    # note -- when building groups, the lower boundary is inclusive, and the upper boundary is non-inclusive: [lower, upper)
    # candidate boundaries = every sample height, except the bottom/top samples, and any heights that are already in required_boundary_heights

    candidate_boundaries = list(heights)

    # remove bottom sample (nothing below), but keep top sample (required to make it a separate cluster since lower bound is inclusive)
    candidate_boundaries = candidate_boundaries[1:]

    # remove boundaries that are already in our 'required' list:

    for h in required_boundary_heights:
        if h in candidate_boundaries:
            candidate_boundaries.remove(h)

    return candidate_boundaries
