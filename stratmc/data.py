import pickle
import sys
import warnings

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.notebook import tqdm

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", ".*The group X_new is not defined in the InferenceData scheme.*")
warnings.filterwarnings("ignore", ".*X_new group is not defined in the InferenceData scheme.*")

def load_data(sample_file, ages_file, proxies = ['d13c'], proxy_sigma_default = 0.1, drop_excluded_samples = True, drop_excluded_ages = True, combine_no_superposition = False):
    """
    Import and pre-process proxy data and age constraints from .csv files formatted according to the :ref:`Data table formatting <datatable_target>` guidelines. To combine data from different .csv files, load each file separately and then combine the DataFrames with :py:meth:`combine_data() <stratmc.data.combine_data>`.

    By default, samples marked ``Exclude? = True`` will be dropped from the data table. If ``sample_file.csv`` includes multiple proxy observations from the same stratigraphic horizon (for a given proxy), then all measurements marked ``Exclude? = False`` and ``superposition? = True ``  will be combined using :py:meth:`combine_duplicates() <stratmc.data.combine_duplicates>`. Samples marked ``superposition? = False`` will remain separate, and their order will be randomized within the inference model. These default behaviors can be modified by passing the ``drop_excluded_samples`` and ``combine_no_superposition`` arguments.

    Parameters
    ----------
    sample_file: str
        Path to .csv file containing proxy data for all sections (without '.csv' extension).

    ages_file: str
        Path to .csv file containing age constraints for all sections (without '.csv' extension).

    proxies: str or list(str), optional
        proxy names (must match column headers in ``sample_file.csv``); defaults to 'd13c'.

    proxy_sigma_default: float or dict{float}, optional
        Measurement uncertainty (:math:`1\\sigma`) to use for proxy observations if not specified in ``proxy_std`` column of ``sample_df``. To set a different value for each proxy, pass a dictionary with proxy names as keys. Defaults to 0.1.

    drop_excluded_samples: bool, optional
        Whether to remove samples with ``Exclude? = True`` from the ``sample_df``; defaults to ``True``. If excluded samples are not dropped, their ages will be passively tracked within the inference model (but they will not be considered during the proxy signal reconstruction).

    drop_excluded_ages: bool, optional
        Whether to remove ages with ``Exclude? = True`` from the ``ages_df``; defaults to ``True``.

    combine_no_superposition: bool, optional
        Whether to combine samples without superposition information by averaging their proxy values; defaults to ``False``.

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

    if 'depositional?' not in list(ages.columns):
        ages['depositional?'] = False

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

    combine_no_superposition: bool, optional
        Whether to combine samples without superposition information by averaging their proxy values; defaults to ``False``.

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

    # concatenating attributes manually -- arviz throws an error when the sampling time attribute (or any other attribute) has a unique value in different traces
    combined_attrs = {}

    attr_keys = list(combined_trace.attrs.keys())

    for key in attr_keys:
        combined_attrs[key] = [combined_trace.attrs[key]]
        del combined_trace.attrs[key]

    del combined_trace.X_new
    for path in tqdm(trace_list[1:]):
        trace = load_trace(path)

        for key in attr_keys:
            combined_attrs[key].append(trace.attrs[key])
            del trace.attrs[key]

        if not np.array_equal(trace.X_new.X_new.values.ravel(), X_new.ravel()):
            sys.exit("Traces have different X_new - check that all inferences were run with the same data and parameters")

        del trace.X_new

        az.concat([combined_trace, trace], dim = 'chain', inplace = True)

    combined_trace.add_groups(dataset)

    combined_trace.attrs = combined_attrs

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
        Location (including the file name, without '.nc' extension) to save ``trace``.

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
        Location (including the file name, without '.pkl' extension) to save ``var``.

    """

    with open(path+'.pkl', "wb") as buff:
        pickle.dump(var, buff)


def load_trace(path):
    """
    Custom load command for NetCDF file containing a trace (:class:`arviz.InferenceData` object saved with :py:meth:`save_trace() <stratmc.data.save_trace>`).

    Parameters
    ----------
    path: str
        Path to saved NetCDF file (without the '.nc' extension).

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
        Path to saved .pkl file (without the '.pkl' extension).

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
        Whether to calculate accumulation rates between every possible sample pairing ('all'), or between successive samples ('successive'); defaults to 'all'.

    age_model: str, optional
        Whether to calculate accumulation rates using the the posterior or prior age model for each section; defaults to 'posterior'.

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

def downsample(sample_df, ages_df, N = 5000, likelihood_ratio_min = 0.5, proxy = 'd13c', keep = 'best', keep_seed = None, resample_with_lowest_n = True, flexible_n = True, best_criteria = 'corr_coef', split_environments = True, **kwargs):
    """
    Subsample a set of proxy observations. Calculates the likelihood of the original stratigraphic signal given the subsampled signal and uncertainty in the data. Returns the solution that meets the mean likelihood ratio minimum with the lowest number of downsampled data points. See input parameter descriptions for additional details.

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    N: int
        Number of random sample groupings to test. Defaults to 5,000.

    likelihood_ratio_min: float or dict{float}, optional
        Minimum acceptable likelihood ratio. For each section, the algorithm selects the smallest downsampled data set that meets this threshold. If multiple solutions with this minimum number of data points exist, then the solution with the highest correlation coefficient is selected if ``keep`` is 'best', while a random one of these solutions is selected if ``keep`` is 'random'. Must be in ``[0, 1]``; defaults to 0.5. Pass as a dictionary to specify a different value for each section.

    keep: str
        If there are multiple solutions that satisfy ``likelihood_ratio_min`` using the minimum possible number of data points, whether to return the best one of these solutions ('best'), or a random solution ('random'). Defaults to 'best'.

    flexible_n: bool
        Whether to consider solutions with 1 more data point than the minimum. Defaults to ``True``.

    resample_with_lowest_n: bool
        Whether to generate another N random solutions with the minimum number of data points required to staisfy ``likelihood_ratio_min`` (or one more than the minimum number of data points, if ``flexible_n = True``). Improves exploration of the solution space. Defaults to ``True``.

    best_criteria: str
        Which metric to use to identify the best solution among the candidate solutions that meet or exceed ``likelihood_ratio_min`` (if ``mode`` is 'best'). Either 'likelihood_ratio' (mean likelihood ratio) or 'corr_coef' (maximum Pearson correlation coefficient); defalts to 'corr_coef'.

    split_environments: bool
        Whether to insert breaks between different depositional environments (using 'Depositional Environment' column in ``sample_df``). Defaults to ``True``.

    proxy: str, optional
        Proxy to downsample. Defaults to 'd13c'.

    sections: list(str) or numpy.array(str), optional
        List of sections to downsample. Defaults to all sections in ``sample_df``.


    Returns
    -------
    downsampled_data: pandas.DataFrame
        :class:`pandas.DataFrame` containing downsampled proxy data. All samples are still included in the DataFrame, but samples that were excluded during downsampling are marked ``Exclude? = True``.

    solution_likelihood_ratios: dict
        Dictionary with the mean likelihood ratio for chosen solutions; keys are section names.

    solution_corr_coefs: dict
        Dictionary with the correlation coefficients for chosen solutions; keys are section names.


    """

    sample_df_downsampled = sample_df.copy()
    sample_df_downsampled['cluster'] = np.nan

    if 'sections' in kwargs:
            sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df_downsampled['section'])

    if type(likelihood_ratio_min) != dict:
        temp = likelihood_ratio_min
        likelihood_ratio_min = {}
        for section in sections:
            likelihood_ratio_min[section] = temp

    solution_likelihood_ratios = {}
    solution_corr_coefs = {}


    for section in tqdm(sections):
        print(f'Downsampling {section}')

        section_df = sample_df[(sample_df['section']==section) & (~sample_df['Exclude?'].astype(bool))].dropna(subset = proxy)

        heights = section_df['height'].values
        proxy_vec = section_df[proxy].values
        proxy_std_vec = section_df[proxy + '_std'].values

        if len(heights) > 2:
            # grab required boundaries (changes in depositional environment, superposition, depositional ages)
            required_boundaries = get_boundaries(sample_df, ages_df, proxy, section, environment = split_environments)

            # make height grid to evaluate likelihood at original heights
            height_grid = heights

            # make proxy grid to evaluate likelihood at original heights
            proxy_grid = proxy_vec
            proxy_std_grid = proxy_std_vec

            min_n_data = len(required_boundaries) - 2 + 1
            max_n_data = len(heights)

            # calculate the likelihood of the original signal, assuming we keep all data points (need for likelihood ratio calculation)
            max_signal_log_prob = np.ones_like(proxy_grid) * np.nan
            for i in range(len(proxy_grid)):
                max_signal_log_prob[i] = norm.logpdf(proxy_grid[i], proxy_grid[i], proxy_std_grid[i]) # mean_proxy_std

            random_data_idx = {}
            corr_coef = np.ones(N) * np.nan
            mean_likelihood_ratio = np.ones(N) * np.nan
            n_data_points = np.ones(N) * np.nan

            for i in tqdm(np.arange(N)):
                rng = np.random.default_rng(seed = i)

                random_data_idx[i] = []
                n_data_points[i] = rng.choice(np.arange(np.max([min_n_data, 2]), max_n_data + 1), 1)[0]

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

                # calculate likelihood of original signal, given interpolated signal and average proxy std dev
                signal_log_prob = np.ones_like(proxy_grid) * np.nan
                for j in range(len(proxy_grid)):
                    # use standard deviation of the measured value, instead of the mean
                    signal_log_prob[j] = norm.logpdf(interp_proxy[j], proxy_grid[j],  proxy_std_grid[j]) # mean_proxy_std

                mean_likelihood_ratio[i] = np.mean(np.exp(signal_log_prob - max_signal_log_prob))

                corr_coef[i] = np.corrcoef(proxy_grid, interp_proxy)[0, 1]

            above = (mean_likelihood_ratio >= likelihood_ratio_min[section])

            keep_idx_list = np.argwhere(above)

            # minimum number of data points in list
            min_n_data = int(np.min(n_data_points[keep_idx_list]))

            # NOTE -- this can rarely cause problems if it doesn't re-find the solution from the previous part w/ the sufficiently high correlation coeficient. in this case, either try re-running w/ higher N, or run with flexible size so that solutions with N+1 samples are also evaluated
            if resample_with_lowest_n:
                print(f'Resampling {section}')
                for i in tqdm(np.arange(N)):

                    rng = np.random.default_rng(seed = i)

                    random_data_idx[i] = []

                    if flexible_n:
                        if min_n_data < len(heights):
                            n_data_points[i] = rng.choice([min_n_data, min_n_data + 1], 1)[0]
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

                    # calculate likelihood of original signal, given interpolated signal and average proxy std dev
                    signal_log_prob = np.ones_like(proxy_grid) * np.nan

                    for j in range(len(proxy_grid)):
                        signal_log_prob[j] = norm.logpdf(interp_proxy[j], proxy_grid[j], proxy_std_grid[j]) # mean_proxy_std

                    mean_likelihood_ratio[i] = np.mean(np.exp(signal_log_prob - max_signal_log_prob))

                    corr_coef[i] = np.corrcoef(proxy_grid, interp_proxy)[0, 1]

            above = (mean_likelihood_ratio >= likelihood_ratio_min[section])

            # solutions that meet minimum criteria
            keep_idx_list = np.argwhere(above)

            if len(keep_idx_list) == 0:
                print('Try running with higher N; no viable solutions found after resampling')

            # if size can be flexible (+1 larger than the minimum) (applies for both 'best' and 'random' modes)
            if flexible_n:
                n_data_idx = np.where((n_data_points[keep_idx_list] == min_n_data) | (n_data_points[keep_idx_list] == min_n_data + 1))[0]

            else:
                # get indices (within candidate list) where number of clusters is equal to the minimum
                n_data_idx = np.where(n_data_points[keep_idx_list] == min_n_data)[0]

            # use user-specified criteria (either corr coef or mean likelihood ratio) to pick the best solution
            if (keep == 'best') or (len(n_data_idx) == 1):
                # out of the viable solutions, choose the one with the highest correlation coefficient
                if best_criteria == 'corr_coef':
                    best_idx_temp = np.nanargmax(corr_coef[keep_idx_list][n_data_idx])

                # out of the viable solutions, choose the one with the lowest (mean) residuals between the interpolated signal and the excluded data points
                elif best_criteria == 'likelihood_ratio':
                    best_idx_temp = np.argmax(mean_likelihood_ratio[keep_idx_list][n_data_idx])

                best_idx = keep_idx_list[n_data_idx[best_idx_temp]][0]

            elif (keep == 'random') and (len(n_data_idx) > 1):
                keep_rng = np.random.default_rng(seed = keep_seed)

                best_idx_temp = keep_rng.choice(n_data_idx, 1)[0]

                best_idx = keep_idx_list[best_idx_temp][0]

            #solution_corr_coefs[section] = corr_coef[best_idx]
            solution_likelihood_ratios[section] = mean_likelihood_ratio[best_idx]
            solution_corr_coefs[section] = corr_coef[best_idx]

            # mark the samples we're keeping as Exclude? = False, and the rest as Exclude? = True
            # don't change samples that didn't have data for the downsampled proxy anyway
            sample_df_downsampled.loc[(sample_df_downsampled['section'] == section) & (~np.isnan(sample_df_downsampled[proxy])), 'Exclude?'] = True
            sample_df_downsampled.loc[random_data_idx[best_idx], 'Exclude?'] = False


    sample_df_downsampled['Exclude?'] = sample_df_downsampled['Exclude?'].astype(bool)

    return sample_df_downsampled, solution_likelihood_ratios, solution_corr_coefs


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
        Whether to insert breaks between different depositional environments (requires 'Depositional Environment' column in ``sample_df``). Defaults to ``True``.

    depositional_ages: bool
        Whether to insert breaks around groups of samples with the same depositional age constraint. Defaults to ``True``.

    superposition:
        Whether to insert breaks around groups of samples without superposition information. Defaults to ``True``.

    Returns
    -------
    boundary_heights: numpy.array
        Array containing required cluster boundaries.

    """

    if 'Depositional Environment' not in list(sample_df.columns):
        sample_df['Depositional Environment'] = np.nan

    sample_df['Depositional Environment'] = sample_df['Depositional Environment'].astype(str)

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
