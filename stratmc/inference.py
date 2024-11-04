import os
import sys
import warnings
from datetime import date
from pathlib import Path
from time import asctime

import arviz as az
import numpy as np
import numpyro
import pandas as pd
import pymc as pm
import pymc.sampling.jax
import xarray as xr
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm

numpyro.enable_x64()

os.environ["PYTENSOR_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float64"

warnings.filterwarnings("ignore", ".*The group X_new is not defined in the InferenceData scheme.*")
warnings.filterwarnings("ignore", ".*X_new group is not defined in the InferenceData scheme.*")


from stratmc.data import clean_data, drop_chains
from stratmc.tests import check_inference


def get_trace(model, gp, ages, sample_df, ages_df, proxies = ['d13c'], approximate = False, name="", chains = 8, draws = 1000, tune = 2000, prior_draws = 1000, target_accept = 0.9, sampler = 'numpyro', nuts_kwargs = None, jitter = 0.001, seed = None, save = True, postprocessing_backend = None, **kwargs):
    """
    Sample the prior and posterior distributions for a :class:`pymc.model.core.Model` returned by :py:meth:`build_model() <stratmc.model.build_model>` in :py:mod:`stratmc.model`. By default, uses :py:func:`pymc.sampling.jax.sample_numpyro_nuts() <pymc.sampling.jax.sample_numpyro_nuts>` to sample the posterior; change ``sampler`` to 'blackjax` to use :py:func:`pymc.sampling.jax.sample_blackjax_nuts() <pymc.sampling.jax.sample_blackjax_nuts>`.

    After the posterior has been sampled, runs :py:meth:`check_inference() <stratmc.tests.check_inference>` in :py:mod:`stratmc.tests` to check that superposition is never violated in the posterior. Any chains with superposition violations are removed from the trace with :py:meth:`drop_chains() <stratmc.data.drop_chains>` before it is returned (if ``save = True``, both the original and 'cleaned' traces are saved to the ``traces`` subfolder), and a warning is issued. See :py:meth:`check_inference() <stratmc.tests.check_inference>` for details; superposition issues are rare, and typically are related to minor violations of detrital or intrusive age constraints.

    Problems during sampling, including frequent divergences or minor violations of limiting age constraints, might be resolved by increasing the number of ``tune`` steps and/or increasing ``target_accept`` (which decreases the step size).

    Parameters
    ----------
    model: PyMC model
        :class:`pymc.model.core.Model` object returned by :py:meth:`build_model() <stratmc.model.build_model>` in :py:mod:`stratmc.model`.

    gp: pymc.gp.Latent
        Gaussian process prior (:class:`pymc.gp.Latent` or :class:`pymc.gp.HSGP`) returned by :py:meth:`build_model() <stratmc.model.build_model>` in :py:mod:`stratmc.model`.

    ages: numpy.array(float)
        array of ages at which to sample the posterior distribution of the proxy signal.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    sections:: list(str) or numpy.array(str), optional
        List of sections to include in the inference model. Defaults to all sections in ``sample_df``.

    proxies: str or list(str), optional
        List of proxies included in the model. Defaults to 'd13c'.

    approximate: bool, optional
        Set to ``True`` if the Hilbert space GP approximation (:class:`pymc.gp.HSGP`) was used in :py:meth:`build_model() <stratmc.model.build_model>`; defaults to ``False``.

    name: str, optional
        Prefix for the saved NetCDF file with the inference results (suffix is timestamp for function call).

    chains: int, optional
        Number of Markov chains to sample in parallel; defaults to 8.

    draws: int, optional
        Number of samples per chain to draw from the posterior; defaults to 1000.

    tune: int, optional
        Number of iterations to tune; defaults to 1000.

    prior_draws: int, optional
        Number of samples to draw from the prior; defaults to 1000.

    target_accept: float, optional
        Between 0 and 1 (exclusive). During tuning, the sampler adapts the proposals such that the average acceptance probability is equal to ``target_accept``; higher values for ``target_accept`` typically lead to smaller step sizes. Defaults to 0.9.

    sampler: str, optional
        Which NUTS algorithm to use to sample the posterior ('numpyro` or 'blackjax`); defaults to 'numpyro`.

    nuts_kwargs: dict, optional
        Dictionary of keyword arguments passed to NumPyro NUTS sampler (see :py:func:`pymc.sampling.jax.sample_numpyro_nuts() <pymc.sampling.jax.sample_numpyro_nuts>` and :class:`numpyro.infer.hmc.NUTS`) or blackjax NUTS sampler (see :py:func:`pymc.sampling.jax.sample_blackjax_nuts() <pymc.sampling.jax.sample_blackjax_nuts>`).

    jitter: float, optional
        Value of ``jitter`` passed to :meth:`pymc.gp.Latent.conditional`. Defaults to 0.001. Changing this value may help if a linear algebra error is encountered during posterior predictive sampling.

    postprocessing_backend: str, optional
        Use the `cpu' or `gpu' for postprocessing. Defaults to ``None``.

    seed: int, optional
        Random seed for sampler.

    save: bool, optional
        Whether to save the trace (to 'traces' subfolder in the current directory); defaults to ``True``.

    Returns
    -------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples.

    """

    if type(proxies) == str:
        proxies = list([proxies])

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    if save:
        # if folders for saving traces don't already exist in the current directory, make them
        dir_path = Path("traces/")
        dir_path_intermediate = Path("traces/temp/")

        if not dir_path.is_dir():
            print('Creating directory for saved traces in ' + str(os.getcwd()))
            dir_path.mkdir()

        if not dir_path_intermediate.is_dir():
            dir_path_intermediate.mkdir()

    if len(asctime().split(" ")[3]) > 1:
        ind = 3
    else:
        ind = 4

    tstamp = (
        str(date.today())
        + "_at_"
        + asctime().split(" ")[ind].replace(":", "_")
    )

    #nuts_dict = {'init_strategy': numpyro.infer.init_to_sample}

    # this version of pymc automatically sets log_likelihood to false if not provided
    idata_kwargs = {'log_likelihood': True}

    with model:
        if sampler == 'numpyro':
            full_trace = pm.sampling.jax.sample_numpyro_nuts(draws, tune=tune, chains=chains, target_accept=target_accept, postprocessing_vectorize='scan', postprocessing_backend = postprocessing_backend, random_seed = seed, idata_kwargs = idata_kwargs, nuts_kwargs = nuts_kwargs)

        elif sampler == 'blackjax':
            full_trace = pm.sampling.jax.sample_blackjax_nuts(draws, tune=tune, chains=chains, target_accept=target_accept, postprocessing_vectorize='scan', postprocessing_backend = postprocessing_backend, random_seed = seed, idata_kwargs = idata_kwargs, nuts_kwargs = nuts_kwargs)

        if save:
            # save the trace with posterior samples -- if an error is encountered during sample_posterior_predictive, can re-load this file so we don't have to start over
            full_trace.to_netcdf("traces/temp/" + str(name) + '_' + tstamp + ".nc")

        v2r = ['f_pred_' + proxy for proxy in proxies]

        for proxy in proxies:
            if approximate:
                f_pred = gp[proxy].conditional('f_pred_' + proxy, Xnew=ages)
            else:
                f_pred = gp[proxy].conditional('f_pred_' + proxy, Xnew=ages, jitter = jitter)

        posterior_predictive = pm.sample_posterior_predictive(
            full_trace,
            var_names=v2r,
            return_inferencedata = True,
            random_seed = seed
        )

        prior = pm.sample_prior_predictive(random_seed = seed, draws = prior_draws)

    full_trace.extend(prior)
    full_trace.extend(posterior_predictive)

    dataset = xr.Dataset()

    dataset = xr.Dataset({"X_new": ("time", ages.ravel())})
    full_trace.add_groups(dataset)

    if save:
        full_trace.to_netcdf("traces/" + str(name) + '_' + tstamp + ".nc")
        # delete the temporary backup
        os.remove("traces/temp/" + str(name) + '_' + tstamp + ".nc")

    bad_chains = check_inference(full_trace, sample_df, ages_df, quiet = True, sections = sections)
    if len(bad_chains) > 0:
        warnings.warn(f"Superposition violated in chains {str(bad_chains)}. These chains were removed from the trace; the original trace (with the bad chains) is saved in the `traces` folder. To investigate the cause of the superposition violation, load the original trace and run the functions in the `stratmc.tests` module with `quiet = False`.")

        full_trace = drop_chains(full_trace, bad_chains)

        # save the clean version
        if save:
            full_trace.to_netcdf("traces/" + 'clean_' + str(name) + '_' + tstamp + ".nc")

    return full_trace


def extend_age_model(full_trace, sample_df, ages_df, new_proxies, new_proxy_df = None, **kwargs):
    """
    Extend age models to a different set of proxy observations using posterior sample ages from an existing inference. For instance, extend an age model built using C isotope data to new S isotope data collected from different stratigraphic horizons in the same sections. Note that the age of stratigraphic horizons that were included in ``sample_df`` (but marked ``Exclude? = True``) is already passively tracked within the model; this function is only required to estimate the age of observations that were not in ``sample_df`` when the inference was run. To estimate ages for new measurements of the same proxy, place the new data in a different column (e.g., 'd13c_new`).

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` with proxy data used during the inference step (as input to :py:meth:`build_model() <stratmc.model.build_model>` in :py:mod:`stratmc.model`).

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints used during the inference step (as input to :py:meth:`build_model() <stratmc.model.build_model>` in :py:mod:`stratmc.model`).

    new_proxies: str or list(str)
        New proxy(s) to construct age models for.

    new_proxy_df: pandas.DataFrame, optional
        :class:`pandas.DataFrame` containing new proxy observations. Optional; if not provided, uses ``sample_df`` (assumes that observations for the new proxy are in the same DataFrame as the original proxy observations).

    sections: list(str) or numpy.array(str), optional
        List of sections included in the inference; only required if not all sections in ``sample_df`` were included.

    Returns
    -------
    interpolated_df: pandas.DataFrame
        :class:`pandas.DataFrame` with interpolated age draws and sample age summary statistics (maximum likelihood estimate, median, and 68% and 95% confidence intervals) for each new proxy observation.
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

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    if type(new_proxies) == str:
        new_proxies = list([new_proxies])

    if new_proxy_df is None:
        new_proxy_df = sample_df.copy()

    # only keep samples that were included in the inference
    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    new_proxy_df, _  = clean_data(new_proxy_df, ages_df, new_proxies, sections)
    new_proxy_df = new_proxy_df[~new_proxy_df['Exclude?']]

    interp_df = pd.DataFrame(columns = list(new_proxy_df.columns) + ['interp'])

    common_sections = [section for section in sections if section in np.unique(new_proxy_df['section'].values)]

    print('Interpolating section age models')
    for section in tqdm(common_sections):
        # height of samples included in inference
        section_df = sample_df[sample_df['section']==section]
        sample_heights = np.concatenate([section_df[~np.isnan(section_df[proxy])]['height'].values for proxy in proxies])

        sample_heights = np.sort(np.unique(sample_heights))

        # heights at which to interpolate age models
        new_section_df = new_proxy_df[new_proxy_df['section']==section]
        new_heights = np.concatenate([new_section_df[~np.isnan(new_section_df[proxy])]['height'].values for proxy in new_proxies])

        new_heights = np.sort(np.unique(new_heights))
        new_heights = np.sort(new_heights)

        # heights of radiometric age constraints
        section_ages_df = ages_df[(ages_df['section']==section) & (~ages_df['Exclude?']) & (~ages_df['intermediate detrital?'])  & (~ages_df['intermediate intrusive?'])]
        age_heights = section_ages_df['height'].values

        # heights at which to interpolate age models
        # just keep all heights, even if included in the original inference - improves workflow for plotting interpolated proxy signals
        interp_heights = [h for h in new_heights] # if h not in sample_heights]
        cols = list(np.unique(list(new_proxy_df.columns)))
        interp_section_df = pd.DataFrame(columns = cols + ['interp'])

        for h in interp_heights:
            idx = new_proxy_df[new_proxy_df['height']==h].index.tolist()
            # this has to be merge because we've added new columns
            interp_section_df = pd.concat([interp_section_df, new_proxy_df.loc[idx]])

        interp_section_df['interp'] = 'y'
        interp_section_df.reset_index(inplace = True, drop = True)

        # sample age posterior - shape (samples x draws)
        sample_age_post = az.extract(full_trace.posterior)[str(section) + '_ages'].values

        age_constraint_post = az.extract(full_trace.posterior)[str(section) + '_radiometric_age'].values

        if sample_age_post.shape[0] != len(sample_heights):
            sys.exit(f"Number of data points for {section} does not match the number of data points in the trace. Check that input data and list of proxies match.")

        if age_constraint_post.shape[0] != len(age_heights):
            sys.exit(f"Number of data points for {section} does not match the number of data points in the trace. Check that input data and list of proxies match.")

        # combine position data for all samples + age constraints included in the inference
        all_heights = np.concatenate([sample_heights, age_heights])
        sorted_idx = np.argsort(all_heights)

        # construct age and height vectors using posteriors for 1) samples in section, and 2) age constraints
        ages_comb = np.vstack([sample_age_post, age_constraint_post])

        age_paths = ages_comb[sorted_idx, :]

        interp_section_ages = np.ones((len((interp_section_df.index.tolist())), age_paths.shape[1])) * np.nan
        # for each posterior draw, extend age model to new stratigraphic horizons
        for i in np.arange(age_paths.shape[1]):
            interp_section_ages[:, i] = np.interp(interp_section_df['height'], all_heights[sorted_idx], age_paths[:, i])


        # store draws for each horizon
        interp_section_df['age_draws'] = np.nan
        interp_section_df['age_draws'] = interp_section_df['age_draws'].astype(object)
        for i in interp_section_df.index.tolist():
            # shape samples x draws
            interp_section_df.at[i, 'age_draws'] = list(interp_section_ages[i, :])

        interp_df = pd.concat([interp_df, interp_section_df])

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
            interp_df.loc[i, 'mle'] = max_like

            # median
            interp_df.loc[i, '50'] = np.percentile(current_ages, 50)

            # 2.5%
            interp_df.loc[i, '2.5'] = np.percentile(current_ages, 2.5)

            # 16%
            interp_df.loc[i, '16'] = np.percentile(current_ages, 16)

            # 84%
            interp_df.loc[i, '84'] = np.percentile(current_ages, 84)

            # 97.5%
            interp_df.loc[i, '97.5'] = np.percentile(current_ages, 97.5)

    return interp_df


def interpolate_proxy(interp_df, proxy, ages):
    """
    Use interpolated sample ages from :py:meth:`extend_age_model() <stratmc.inference.extend_age_model>` to calculate proxy values at a given set of ages (e.g., to plot 68 and 95% confidence intervals over time for a new proxy using :py:meth:`interpolated_proxy_inference() <stratmc.plotting.interpolated_proxy_inference>` in :py:mod:`stratmc.plotting`).

    Parameters
    ----------
    interp_df: pandas.DataFrame
        :class:`pandas.DataFrame` with proxy data and interpolated ages from :py:meth:`extend_age_model() <stratmc.inference.extend_age_model>`.

    proxy: str
        Tracer to interpolate.

    ages: list(float) or numpy.array(float)
        Target ages at which to interpolate proxy values.

    Returns
    -------
    interpolated_proxy_df: pandas.DataFrame
        :class:`pandas.DataFrame` with interpolated proxy values and summary statistics (maximum likelihood estimate, median, and 68% and 95% confidence intervals) at each age in ``ages``.
    """

    age_paths = np.vstack(np.asarray(interp_df['age_draws']))

    proxy_vec = interp_df[proxy]

    # iterate over draws
    print('Interpolating proxy values')
    for i in tqdm(np.arange(age_paths.shape[1])):
        current_order = np.argsort(age_paths[:, i])
        current_path = age_paths[current_order, i]
        current_interp_proxy = np.interp(ages, current_path, proxy_vec[current_order])

        if i == 0:
                interp_proxy = current_interp_proxy

        else:
            interp_proxy = np.vstack((interp_proxy, current_interp_proxy))

    interpolated_proxy_df = pd.DataFrame(columns = ['age', proxy + '_draws', 'mle', '2.5', '16', '50', '84', '97.5'])
    interpolated_proxy_df[proxy + '_draws'] = interpolated_proxy_df[proxy + '_draws'].astype(object)


    # iterate over ages
    print('Calculating summary statistics')
    for i in tqdm(np.arange(len(ages))):
        # shape = draws x ages
        current_proxy = interp_proxy[:, i]
        dy = np.linspace(np.min(current_proxy), np.max(current_proxy), 200)
        max_like = dy[np.argmax(gaussian_kde(current_proxy, bw_method = 1)(dy))]
        current_mle = max_like

        temp_df = pd.DataFrame({'age': ages[i],
                                proxy + '_draws': [current_proxy],
                                'mle': current_mle,
                                '2.5': np.percentile(current_proxy, 2.5),
                                '16': np.percentile(current_proxy, 16),
                                '50': np.percentile(current_proxy, 50),
                                '84': np.percentile(current_proxy, 84),
                                '97.5': np.percentile(current_proxy, 97.5),
                               })

        interpolated_proxy_df = pd.concat([interpolated_proxy_df, temp_df])

    return interpolated_proxy_df


def age_range_to_height(full_trace, sample_df, ages_df, lower_age, upper_age, **kwargs):
    """
    Use the posterior age model for each section to find the stratigraphic interval (with uncertainty) corresponding to a given age range. If ``sections`` is not provided, returns height estimates for every section that overlaps the target age range. To visualize the stratigraphic intervals, use :py:meth:`section_age_range() <stratmc.plotting.section_age_range>` in :py:mod:`stratmc.plotting`.

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    lower_age: float
        Lower bound (youngest age) of the target age interval.

    upper_age: float
        Upper bound (oldest age) of the target age interval.

    sections: list(str) or numpy.array(str), optional
        List of sections included in the original inference; only required if not all sections in ``sample_df`` were included.

    Returns
    -------
    height_range_df: pandas.DataFrame
        Summary statistics for the base and top height of the target age interval (maximum likelihood estimate, median, and 68% and 95% confidence intervals) for each section.

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

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    ages_new = full_trace.X_new.X_new.values
    above = ages_new >= lower_age
    below = ages_new <= upper_age
    target_ind = np.where(above & below)[0]
    age_target = ages_new[target_ind]

    height_paths = {}
    keep_sections = []
    print('Mapping age models to sections')
    for section in tqdm(sections):
        # all sample + age constraint heights
        section_df = sample_df[sample_df['section']==section]
        sample_heights = np.concatenate([section_df[~np.isnan(section_df[proxy])]['height'].values for proxy in proxies])

        sample_heights = np.unique(sample_heights)
        sample_heights = np.sort(sample_heights)

        # heights of radiometric age constraints
        section_ages_df =  ages_df[(ages_df['section']==section) & (~ages_df['Exclude?']) & (~ages_df['intermediate detrital?'])  & (~ages_df['intermediate intrusive?'])]
        age_heights = section_ages_df['height'].values

        # sample age posterior - shape (samples x draws)
        sample_age_post = az.extract(full_trace.posterior)[str(section) + '_ages'].values

        age_constraint_post = az.extract(full_trace.posterior)[str(section) + '_radiometric_age'].values

        # combine position data for all samples + age constraints included in the inference
        all_heights = np.concatenate([sample_heights, age_heights])
        sorted_idx = np.argsort(all_heights)
        all_heights_sort = all_heights[sorted_idx]

        lowest_age = np.min(sample_age_post.ravel())
        highest_age = np.max(sample_age_post.ravel())

        # only include section if it overlaps the target age range
        target_range = pd.Interval(np.min(age_target), np.max(age_target), closed = 'both')
        section_range = pd.Interval(lowest_age, highest_age, closed = 'both')
        result = target_range.overlaps(section_range)

        if result:
            keep_sections.append(section)

            # construct age and height vectors for the current draw using posteriors for 1) samples in section, and 2) age constraints
            for i in np.arange(sample_age_post.shape[1]):
                sample_age_vec = sample_age_post[:, i]
                constraint_age_vec = age_constraint_post[:, i]
                age_vec = np.concatenate([sample_age_vec, constraint_age_vec])
                age_vec_sort = np.flip(age_vec[sorted_idx]) # young to old (low to high)

                # interpolate - x is age (must be strictly increasing), y is age
                interp_heights = np.interp(age_target, age_vec_sort, np.flip(all_heights_sort.astype(float))) # young/high to old/low
                interp_heights = np.asarray(interp_heights).reshape(len(age_target), 1)

                if i == 0:
                    height_paths[section] = interp_heights

                else:
                    height_paths[section] = np.hstack((height_paths[section], interp_heights))


    # dataframe with: max likelihood, 2.5, 16, 50, 84, 97.5 percentiles for the top and bottom of the target interval for each section
    height_range_df = pd.DataFrame(columns = ['section', 'base_mle', 'base_2.5', 'base_16', 'base_50', 'base_84', 'base_97.5',
                                      'top_mle', 'top_2.5', 'top_16', 'top_50', 'top_84', 'top_97.5'])

    for section in keep_sections:
        section_stats = {}
        section_stats['section'] = section
        section_df = sample_df[sample_df['section']==section]

        # 0 is the top height, -1 is the bottom height
        section_stats['top_2.5'] = np.percentile(height_paths[section][0], 2.5)
        section_stats['base_2.5'] = np.percentile(height_paths[section][-1], 2.5)
        section_stats['top_97.5'] = np.percentile(height_paths[section][0], 97.5)
        section_stats['base_97.5'] = np.percentile(height_paths[section][-1], 97.5)

        section_stats['top_16'] = np.percentile(height_paths[section][0], 16)
        section_stats['base_16'] = np.percentile(height_paths[section][-1], 16)
        section_stats['top_84'] = np.percentile(height_paths[section][0], 100-16)
        section_stats['base_84'] = np.percentile(height_paths[section][-1], 100-16)

        section_stats['top_50'] = np.percentile(height_paths[section][0], 50)
        section_stats['base_50'] = np.percentile(height_paths[section][-1], 50)

        # only calculate MLE if there's more than 1 unique value (edge case where interpolations all hit top/bottom of section)
        if len(np.unique(height_paths[section][-1])) > 1:
            dy = np.linspace(np.min(height_paths[section][-1]), np.max(height_paths[section][-1]), 200)
            section_stats['base_mle'] = dy[np.argmax(gaussian_kde(height_paths[section][-1], bw_method = 1)(dy))]
        else:
            section_stats['base_mle'] = np.unique(height_paths[section][-1])

        if len(np.unique(height_paths[section][0])) > 1:
            dy = np.linspace(np.min(height_paths[section][0]), np.max(height_paths[section][0]), 200)
            section_stats['top_mle'] = dy[np.argmax(gaussian_kde(height_paths[section][0], bw_method = 1)(dy))]

        else:
            section_stats['top_mle'] = np.unique(height_paths[section][0])

        section_stats_df = pd.DataFrame(section_stats, index = [0])
        height_range_df = pd.concat([height_range_df.astype(section_stats_df.dtypes), section_stats_df], ignore_index = True)

    height_range_df.set_index('section', inplace = True)

    return height_range_df

def map_ages_to_section(full_trace, sample_df, ages_df, include_radiometric_ages = False, **kwargs):
    """
    Helper function for :py:meth:`section_proxy_signal() <stratmc.plotting.section_proxy_signal>` in :py:mod:`stratmc.plotting`. Maps the ``ages`` array passed to  :py:meth:`get_trace() <stratmc.inference.get_trace>` to height in each section using the most likely posterior age models.

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    include_radiometric_ages: bool, optional
        Whether to consider radiometric ages when calculating the most likely posterior age model for each section; defaults to ``False``.

    sections: list(str) or numpy.array(str), optional
        List of sections included in the inference; only required if not all sections in ``sample_df`` were included.

    Returns
    -------
    age_model_df: pandas.DataFrame
        :class:`pandas.DataFrame` with interpolated heights at each age in the ``ages`` vector that was passed to :py:meth:`get_trace() <stratmc.inference.get_trace>`.

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

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    # only keep samples that were included in the inference
    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    ages_new = full_trace.X_new.X_new.values

    age_model_df = pd.DataFrame(columns = ['section', 'age', 'interpolated height'])

    # get posterior age model for each section
    age_paths = {}
    for section in sections:
        # all sample + age constraint heights
        section_df = sample_df[sample_df['section']==section]

        sample_heights = section_df['height'].values

        sample_heights = np.sort(np.unique(sample_heights))

        # heights of radiometric age constraints

        section_ages_df = ages_df[(ages_df['section']==section) & (~ages_df['Exclude?']) & (~ages_df['intermediate detrital?'])  & (~ages_df['intermediate intrusive?'])]
        age_heights = section_ages_df['height'].values

        # sample age posterior - shape = (samples x draws)
        sample_age_post = az.extract(full_trace.posterior)[str(section) + '_ages'].values

        # get posterior age draws for section
        if include_radiometric_ages:
            age_constraint_post = az.extract(full_trace.posterior)[str(section) + '_radiometric_age'].values

            # combine position data for all samples + age constraints included in the inference
            all_heights = np.concatenate([sample_heights, age_heights])
            sorted_idx = np.argsort(all_heights)
            all_heights_sort = all_heights[sorted_idx]

            ages_comb = np.vstack([sample_age_post, age_constraint_post])

            age_paths[section] = ages_comb[sorted_idx, :]

        else:
            sorted_idx = np.argsort(sample_heights)
            all_heights_sort = sample_heights[sorted_idx]

            age_paths[section] = sample_age_post[sorted_idx, :]

        # calculate MLE age model
        # shape of age_paths: (observations, draws)
        max_like = np.zeros(age_paths[section].shape[0])
        for i in np.arange(age_paths[section].shape[0]):
            sample_ages = age_paths[section][i, :]
            dx = np.linspace(np.min(sample_ages), np.max(sample_ages), 1000)
            max_like[i] = dx[np.argmax(gaussian_kde(sample_ages, bw_method = 1)(dx))]

        lowest_age = np.min(max_like)
        highest_age = np.max(max_like)

        below = ages_new >= lowest_age
        above = ages_new <= highest_age
        target_age_ind = np.where(below & above)[0]

        target_age_vec = ages_new[target_age_ind]

        # interpolate the MLE age model
        interp_height = np.interp(target_age_vec, np.flip(max_like), np.flip(all_heights_sort).astype(np.float64))
        section_df_temp = pd.DataFrame({'section': [section] * interp_height.shape[0],
                                          'age': target_age_vec,
                                          'interpolated height': interp_height,
                                          })

        age_model_df = pd.concat([age_model_df, section_df_temp], ignore_index = True)

    return age_model_df


def count_samples(full_trace, time_grid = None):
    """
    Helper function for  :py:meth:`proxy_data_density() <stratmc.plotting.proxy_data_density>` in :py:mod:`stratmc.plotting`. Counts the number of observations within discrete time bins (based on the posterior sample ages).

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    time_grid: np.array, optional
        Time bin edges; if not provided, defaults to the ``ages`` array passed to :py:meth:`get_trace() <stratmc.inference.get_trace>`.

    Returns
    -------
    sample_counts: np.array
        Number of observations in each time bin, summed over all posterior draws such that the average number of observations is ``sample_counts/n``.

    grid_centers: np.array
        Time bin centers.

    grid_widths: np.array
        Time bin widths.

    n: int
        Number of posterior draws in ``full_trace``.
    """

    sample_ages_post = az.extract(full_trace.posterior)['ages'].values

    if time_grid is None:
        time_grid = full_trace.X_new.X_new.values

    # for each grid point
    sample_counts = np.zeros(len(time_grid) - 1)
    grid_centers = np.diff(time_grid)/2 + time_grid[:-1]
    grid_widths = np.diff(time_grid)

    print('Counting data in time bins')
    for i in tqdm(np.arange(len(time_grid) - 1)):
        for j in np.arange(sample_ages_post.shape[1]):
            n_overlap =  np.argwhere(((sample_ages_post[:, j] >= time_grid[i]) & (sample_ages_post[:, j] < time_grid[i + 1]))).shape[0]
            sample_counts[i] += n_overlap

    n = sample_ages_post.shape[1]

    return sample_counts, grid_centers, grid_widths, n

def find_gaps(full_trace, time_grid = None):
    """
    Helper function for  :py:meth:`proxy_data_gaps() <stratmc.plotting.proxy_data_gaps>` in :py:mod:`stratmc.plotting`. Counts the number of draws from the posterior where there are no observations within discrete time bins (based on the posterior sample ages).

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    time_grid: np.array, optional
        Time bin edges; if not provided, defaults to the ``ages`` array passed to :py:meth:`get_trace() <stratmc.inference.get_trace>`.

    Returns
    -------
    age_gaps: np.array of int
        Number of posterior draws where there are no observations; each entry corresponds to an age bin (corresponding to ``grid_centers`` and ``grid_widths``).

    grid_centers: np.array
        Time bin centers.

    grid_widths: np.array
        Time bin widths.

    n: int
        Number of posterior draws in ``full_trace``.

    """

    sample_ages_post = az.extract(full_trace.posterior)['ages'].values

    if time_grid is None:
        time_grid = full_trace.X_new.X_new.values

    # for each grid point
    age_gaps = np.zeros(len(time_grid) - 1)
    grid_centers = np.diff(time_grid)/2 + time_grid[:-1]
    grid_widths = np.diff(time_grid)

    print('Calculating gaps in the data')
    for i in tqdm(np.arange(len(time_grid) - 1)):
        for j in np.arange(sample_ages_post.shape[1]):
            if all((sample_ages_post[:, j] <= time_grid[i]) | (sample_ages_post[:, j] > time_grid[i + 1])):
                age_gaps[i] += 1

    n = sample_ages_post.shape[1]

    return age_gaps, grid_centers, grid_widths, n


def calculate_lengthscale_stability(full_trace, **kwargs):

    """
    Compute the posterior standard deviation of the :class:`pymc.gp.cov.ExpQuad <pymc.gp.cov.ExpQuad>` covariance kernel lengthscale as additional chains are considered (i.e., for 1 to `N` chains). When the posterior has been sufficiently explored, the standard deviation will stabilize; if it has not stabilized, then additional chains should be run. Helper function for :py:meth:`lengthscale_stability() <stratmc.plotting.lengthscale_stability>` in :py:mod:`stratmc.plotting`.

    To consider chains from multiple traces associated with the same inference model, first combine the traces (saved as NetCDF files) using :py:meth:`combine_traces() <stratmc.data.combine_traces>` in :py:mod:`stratmc.data`.

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    proxy: str, optional
        Name of the proxy; only required if multiple proxies were included in the inference model.

    Returns
    -------
    gp_ls_std: np.array of float
        Posterior standard deviation of the covariance kernel lengthscale posterior; entries correspond to number of chains considered (first entry is 1 chain, last entry is all `N` chains).

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

    gp_ls_post = full_trace.posterior['gp_ls_' + str(proxy)].values

    gp_ls_std = []

    # calculate variance when considering 1 to N chains
    for i in np.arange(gp_ls_post.shape[0]):
        gp_ls_std.append((np.std(gp_ls_post[0:i + 1, :, :])))

    return gp_ls_std


def calculate_proxy_signal_stability(full_trace, **kwargs):

    """
    Compute the residuals between the median inferred proxy signal when all *N* chains are considered compared to when 1 to *N-1* chains are considered. When the posterior has been sufficiently explored, the residuals will stabilize and approach zero; if they have not stabilized, then additional chains should be run. Helper function for :py:meth:`proxy_signal_stability() <stratmc.plotting>` in :py:mod:`stratmc.plotting`.

    To consider chains from multiple traces associated with the same inference model, first combine the traces (saved as NetCDF files) using :py:meth:`combine_traces() <stratmc.data.combine_traces>` in :py:mod:`stratmc.data`.

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    proxy: str, optional
        Name of the proxy; only required if multiple proxies were included in the inference model.

    Returns
    -------
    median_residuals: np.array of float
        Residuals between the median inferred proxy value (at each time in ``ages`` passed to :py:meth:`get_trace() <stratmc.inference.get_trace>`) calculated using 1 to `N-1` chains versus all `N` chains. Shape is (chains x ages).

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

    post_proxy = full_trace.posterior_predictive['f_pred_' + proxy].values

    median_proxy = []

    for i in np.arange(post_proxy.shape[0]):
        selected_chains = np.arange(0, i + 1)
        current_data = post_proxy[selected_chains, :, :].reshape(len(selected_chains) * post_proxy.shape[1], post_proxy.shape[2])
        median_proxy.append(np.median(current_data, axis = 0))

    median_residuals = []
    for i in np.arange(post_proxy.shape[0]):
        median_residuals.append(np.abs(median_proxy[-1] - median_proxy[i]))

    median_residuals = np.array(median_residuals)

    return median_residuals
