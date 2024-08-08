import random
import sys
import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", ".*The group X_new is not defined in the InferenceData scheme.*")
warnings.filterwarnings("ignore", ".*X_new group is not defined in the InferenceData scheme.*")

pd.options.mode.chained_assignment = None

from stratmc.model import build_model


def make_excursion(time, amplitude, baseline = 0, rising_time = None, rate_offset = True, excursion_duration = None, min_duration = 1,
                    smooth = False, smoothing_factor = 10, seed = None):

    """
    Function for generating a synthetic proxy signal that contains a number of user-specified excursions.

    Parameters
    ----------
    time: numpy.array(float)
        Time vector over which to generate proxy signal.

    amplitude: float, list(float), or numpy.array(float)
        Amplitude of excursion; pass a list or array to generate multiple excursions.

    baseline: float, optional
        Baseline proxy value. Defaults to 0.

    rising_time: float, list(float), or numpy.array(float), optional
        Fraction of excursion duration spent on the rising limb (linear increase/decrease toward peak).
        Must be between 0 and 1. If not provided, randomly generated if ``rate_offset`` is ``True`` and set to 0.5 if ``rate_offset`` is  ``False``. Pass a list to specify different rising times for each excursion.

    rate_offset: bool, optional
        If ``False``, rising and falling limbs of excursion have equal duration. If ``True``, the fraction of the excursion duration spent on the rising limb is set by ``rising_time``. Defaults to ``False``.

    excursion_duration: float, list(float), or numpy.array(float), optional
        Duration of excursion; pass a list or array to generate multiple excursions. Random if not provided.

    min_duration: float, optional
        Minimum excursion duration if ``excursion_duration`` is not provided. Defaults to 1.

    smooth: bool, optional
        Whether to smooth excursion peaks. Defaults to ``False``.

    smoothing_factor: float, optional
        Smoothing factor if ``smooth`` is ``True``; higher values produce smoother signals. Defaults to 10.

    seed: int, optional
        Random seed used to generate signal.

    Returns
    -------
    interp_proxy: np.array
        Tracer signal interpolated to points in the ``time`` vector
    """

    if seed is not None:
        random.seed(a = seed)
        np.random.seed(seed)

    if (type(amplitude) == float) or (type(amplitude) == int):
        amplitude = np.array([amplitude])

    else:
        amplitude = np.array(amplitude)

    # if excursion duration not in inputs
    if excursion_duration == None:
        if len(amplitude) == 1:
            excursion_duration = random.uniform(np.min(time), np.max(time)-np.min(time))
            excursion_duration = np.asarray([excursion_duration])
            n = 1

        # multiple excursions
        else:
            excursion_duration = np.asarray([excursion_duration])
            amplitude = np.asarray(amplitude)
            n = len(amplitude)
            excursion_duration = []
            duration_sum = 0
            for i in np.arange(len(amplitude)).tolist():
                max_duration = np.max(time) - np.min(time) - duration_sum - (min_duration * (len(amplitude)-i))
                excursion_duration.append(random.uniform(min_duration, max_duration))
                duration_sum = np.sum(excursion_duration)

        excursion_duration = np.array(list(excursion_duration))

    # if excursion duration in inputs
    else:
        excursion_duration = np.array(list(excursion_duration))
        n = len(amplitude)
        if len(amplitude) != len(excursion_duration):
            sys.exit('Duration and amplitude lists are not the same length')


    # fraction of excursion duration spent on rising vs. falling limb of excursion
    if (rising_time == None) and (rate_offset):
        rising_frac = np.random.uniform(low = 0.1, high = 0.9, size = len(excursion_duration))

    elif (rising_time == None) and (not rate_offset):
        rising_frac = np.ones(len(excursion_duration)) * 0.5

    else:
        rising_frac = rising_time

    rising_frac = np.asarray(rising_frac)

    # if more than 1 excursion, choose random starting point for each excursion
    if len(excursion_duration) > 1:
        # check that total excursion duration does not exceed total timespan
        if np.sum(excursion_duration) > np.max(time) - np.min(time):
            sys.exit('Sum of excursion durations exceeds total time')

        excursion_start = []
        current_min = np.min(time)
        for i in np.arange(len(excursion_duration)):
            max_start = np.max(time) - np.sum(excursion_duration[i:])
            start = random.uniform(current_min, max_start)
            excursion_start.append(start)
            current_min = excursion_start[-1] + excursion_duration[i]

    # if only 1 excursion, choose random starting point
    else:
        excursion_start = [random.uniform(np.min(time), np.max(time) - excursion_duration)]
        excursion_start = np.asarray(excursion_start)

    # generate proxy signal for each excursion
    excursion_proxy = {}
    excursion_time = {}
    if len(excursion_duration) > 1:
        for i, amp, dur, start in zip(np.arange(len(excursion_duration)), amplitude, excursion_duration, excursion_start):
            peak_time = start + (rising_frac[i] * dur)
            base_time = start
            end_time = start + dur

            # fit line through rising limb
            rising_p = np.polyfit([base_time, peak_time], [0, amp], 1)

            rising_t = np.linspace(base_time, peak_time, 1000)
            rising_proxy = np.polyval(rising_p, rising_t)

            # fit line through falling limb
            falling_p = np.polyfit([peak_time, end_time], [amp, 0], 1)

            falling_t = np.linspace(peak_time+1e-6, end_time, 1000)
            falling_proxy = np.polyval(falling_p, falling_t)

            # combine rising and falling limbs
            excursion_proxy[i] = np.concatenate([rising_proxy, falling_proxy])
            excursion_time[i] = np.concatenate([rising_t, falling_t])

            if smooth:
                spline = UnivariateSpline(excursion_time[i], excursion_proxy[i], s = smoothing_factor)
                x_new = np.linspace(np.min(excursion_time[i]), np.max(excursion_time[i]), 2000)
                excursion_proxy[i] = spline(x_new)
                excursion_time[i] = x_new
    else:
        peak_time = excursion_start[0] + (rising_frac * excursion_duration[0])
        base_time = excursion_start[0]
        end_time = excursion_start[0] + excursion_duration[0]

        # fit line through rising limb
        rising_p = np.polyfit([base_time[0], peak_time[0]], [0, amplitude[0]], 1)

        rising_t = np.linspace(base_time[0], peak_time[0], 1000)
        rising_proxy = np.polyval(rising_p, rising_t)

        # fit line through falling limb
        falling_p = np.polyfit([peak_time[0], end_time[0]], [amplitude[0], 0], 1)

        falling_t = np.linspace(peak_time[0]+1e-6, end_time[0], 1000)
        falling_proxy = np.polyval(falling_p, falling_t)

        # combine rising and falling limbs
        excursion_proxy[0] = np.concatenate([rising_proxy, falling_proxy])
        excursion_time[0] = np.concatenate([rising_t, falling_t])

        if smooth:
            spline = UnivariateSpline(excursion_time[0], excursion_proxy[0], s = smoothing_factor)
            x_new = np.linspace(np.min(excursion_time[0]), np.max(excursion_time[0]), 2000)
            excursion_proxy[0] = spline(x_new)
            excursion_time[0] = x_new


    # fill with baseline outside of excursions
    timeline = []
    proxy = []
    for i in np.arange(n):
        if i == 0 and excursion_start[i] > 0:
            s = np.linspace(0, excursion_start[i], 1000)
            d = np.zeros(len(s))
            timeline = np.insert(timeline, 0, s.ravel())

            proxy = np.insert(proxy, 0, d)

        timeline = np.append(timeline, excursion_time[i])
        proxy = np.append(proxy, excursion_proxy[i])

        if i < len(excursion_duration)-1:
            if np.max(excursion_time[i]) < np.min(excursion_time[i+1]):
                f = np.linspace(np.max(excursion_time[i]) + 1e-3, np.min(excursion_time[i+1]), 1000)
                timeline = np.append(timeline, f)

                d = np.zeros(len(f))
                proxy = np.append(proxy, d)

        if i == len(excursion_duration)-1:
            if timeline[-1] < np.max(time):
                f = np.linspace(timeline[-1]+1e-3, np.max(time), 1000)
                timeline = np.append(timeline, f)

                d = np.zeros(len(f))
                proxy = np.append(proxy, d)

    # interpolate to original time vector
    proxy_fun = interp1d(timeline, proxy)
    interp_proxy = proxy_fun(time) + baseline

    return interp_proxy

def synthetic_sections(true_time, true_proxy, num_sections, num_samples, max_section_thickness, proxies = ['d13c'], noise = False, noise_amp = 0.1, min_constraints = 2, max_constraints = 3, seed = None, **kwargs):

    """
    Function for generating synthetic proxy observations and age constraints using a predefined proxy signal.

    Parameters
    ----------
    true_time: numpy.array(float)
        True time vector for input signal.

    true_proxy: numpy.array(float) or dict{numpy.array(float)}
        True proxy vector for input signal. If generating synthetic data for multiple proxies, pass as a dictionary with proxy names as keys.

    num_sections: int
        Number of synthetic sections to generate.

    num_samples: int
        Number of samples per synthetic section.

    max_section_thickness: float
        Maximum thickness of synthetic sections.

    proxies: str or list(str), optional
        Column name(s) for synthetic proxy observations in ``sample_df``. Defaults to 'd13c'.

    noise: bool, optional
        Whether to add white noise to proxy observations. Defaults to ``False``.

    noise_amp: float or dict{float}, optional
        Amplitude of white noise added to proxy observations (if ``noise`` is ``True``). To specify a different noise amplitude for each proxy, pass as a dictionary with proxy names as keys. Defaults to 0.1.

    min_constraints: int, optional
        Minimum number of age constraints per synthetic section (must be at least 2). Defaults to 2.

    max_constraints: int, optional
        Maximum number of age constraints per synthetic section. Defaults to 3.

    seed: int, optional
        Random seed used to generate synthetic sections.

    Returns
    -------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for synthetic sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for synthetic sections.
    """

    if type(proxies) == str:
        proxies = [proxies]

    if type(true_proxy) != dict:
        temp = true_proxy
        true_proxy = {}
        for proxy in proxies:
            true_proxy[proxy] = temp

    section_ages = []

    proxy_vec = {}
    for proxy in proxies:
        proxy_vec[proxy] = []

    heights = []
    age_heights = []
    ages = []
    age_std = []
    age_section_names = []
    section_names = []

    for i in np.arange(num_sections):
        proxy_temp = {}

        section = str(i)

        if seed is not None:
            random.seed(a = int(seed+i))
            np.random.seed(seed+i)

        if 'age_constraints' in kwargs:
            # ages
            ages_dict = kwargs['age_constraints']
            ages_std_dict = kwargs['age_constraints_std']
            section_ages_temp = np.flip(np.sort(np.random.uniform(np.min(ages_dict[section]), np.max(ages_dict[section]), num_samples)))

        else:
            # sample ages
            section_ages_temp = np.flip(np.sort(np.random.uniform(np.min(true_time), np.max(true_time), num_samples)))

            # true_time must be strictly increasing for valid interpolation
        if not np.all(np.diff(np.flip(true_time)) > 0):
            for proxy in proxies:
                true_proxy[proxy] = true_proxy[proxy][np.argsort(true_time)]
            true_time = np.sort(true_time)

        # sample proxy
        for proxy in proxies:
            proxy_temp[proxy] = np.interp(section_ages_temp, true_time, true_proxy[proxy])


        # sample heights
        heights_temp = np.random.uniform(0.5, max_section_thickness - np.random.choice(np.random.uniform(0, 0.75 * max_section_thickness, 100000), size = 1, replace = False), num_samples)

        heights_temp = np.sort(heights_temp)

        if 'age_constraints' in kwargs:
            ages_temp = ages_dict[section]
            sort_idx = np.argsort(ages_temp)
            ages_temp = np.flip(ages_temp[sort_idx])
            age_std_temp = ages_std_dict[section]
            age_std_temp = np.flip(age_std_temp[sort_idx])

            age_heights_temp = np.flip(np.interp(ages_temp, np.flip(section_ages_temp), np.flip(heights_temp)))
            if age_heights_temp[0] >= heights_temp[0]:
                age_heights_temp[0] = 0
            if age_heights_temp[-1] <= heights_temp[-1]:
                age_heights_temp[-1] = heights_temp[-1] + 1

        else:
            # number of age constraints
            num_ages = random.randrange(min_constraints,max_constraints)#random.randrange(2,3)

            # height of age constraints
            age_heights_temp = np.sort(np.random.choice(np.random.uniform(0, np.max(heights_temp), 100000), replace = False, size = num_ages))

            if age_heights_temp[0] > heights_temp[0]:
                age_heights_temp = np.append(age_heights_temp, 0)
                num_ages += 1

            if age_heights_temp[-1] < heights_temp[-1]:
                age_heights_temp = np.append(age_heights_temp, heights_temp[-1]+np.absolute(np.random.normal(3, 1, 1)))
                num_ages += 1

            age_heights_temp.sort()

            # age of constraints
            ages_temp = np.interp(age_heights_temp, heights_temp, section_ages_temp)

            # age uncertainties
            age_std_temp = abs(np.random.normal(0.5, 0.5, num_ages))


        age_section_names_temp = [str(i)]*len(age_heights_temp)
        section_names_temp = [str(i)]*len(heights_temp)

        section_ages = np.append(section_ages, section_ages_temp)


        if noise:
            if type(noise_amp) == float:
                temp = noise_amp
                noise_amp = {}
                for proxy in proxies:
                    noise_amp[proxy] = np.ones(num_sections) * temp

            elif (type(noise_amp) == dict):
                if (type(noise_amp[proxies[0]]) == float) | (type(noise_amp[proxies[0]]) == int):
                    for proxy in proxies:
                        n = np.random.normal(0, noise_amp[proxy], len(proxy_temp[proxy]))
                        proxy_temp[proxy] = proxy_temp[proxy] + n

                else:
                    for proxy in proxies:
                        n = np.random.normal(0, noise_amp[proxy][i], len(proxy_temp[proxy]))
                        proxy_temp[proxy] = proxy_temp[proxy] + n

        for proxy in proxies:
            proxy_vec[proxy] = np.append(proxy_vec[proxy], proxy_temp[proxy])

        heights = np.append(heights, heights_temp)

        ages = np.append(ages, ages_temp)
        age_heights = np.append(age_heights, age_heights_temp)
        age_std = np.append(age_std, age_std_temp)

        age_section_names = np.append(age_section_names, age_section_names_temp)
        section_names = np.append(section_names, section_names_temp)


    ages_df, sample_df = synthetic_signal_to_df(proxy_vec,
                                                heights,
                                                section_ages,
                                                section_names,
                                                ages, age_std,
                                                age_heights,
                                                age_section_names,
                                                proxies = proxies)

    return ages_df, sample_df

def synthetic_signal_to_df(proxy_vec, heights, section_ages, section_names, ages, age_std, age_heights, age_section_names, proxies = ['d13c']):

    """
    Helper function for generating artificial sample and age data using :py:meth:`synthetic_sections() <stratmc.synthetics>`.

    Parameters
    ----------
    proxy_vec: np.array(float) or dict{np.array(float)}
        Array of proxy observations. Pass as a dictionary if more than one proxy.

    heights: np.array(float)
        Array of heights corresponding to proxy observations in ``proxy_vec``.

    section_ages: np.array(float)
        Array of ages corresponding to proxy observations in ``proxy_vec``.

    section_names: np.array(str)
        Array of section names corresponding to proxy observations in ``proxy_vec``.

    ages: np.array(float)
        Array of age constraints.

    age_std: np.array(float)
        Array of uncertainties for each age constraint in ``ages``.

    age_heights: np.array(float)
        Array of heights for each age constraint in ``ages``.

    age_section_names: np.array(str)
        Array of section names corresponding to age constraints in ``ages``.

    proxies: str or list(str), optional
        Name(s) of proxies. Defaults to `d13c`.

    Returns
    -------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for synthetic sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for synthetic sections.
    """

    if type(proxies) == str:
        proxies = [proxies]

    if type(proxy_vec) != dict:
        temp = proxy_vec
        proxy_vec = {}
        proxy_vec[proxies[0]] = temp

    # ages dataframe:
    ages_df = {'section': age_section_names,
               'height': age_heights,
              'age': ages,
              'age_std': age_std}

    ages_df = pd.DataFrame(data = ages_df)

    ages_df.sort_values(by = ['section', 'height'], inplace = True)

    # sample dataframe:
    sample_df = {'section': section_names,
                 'height': heights,
                 'age': section_ages,
                 }

    for proxy in proxies:
        sample_df[proxy] = proxy_vec[proxy]

    sample_df = pd.DataFrame(data = sample_df)

    sample_df.sort_values(by = ['section', 'height'], inplace = True)

    ages_df['shared?'] = False
    ages_df['name'] = np.nan
    ages_df['intermediate detrital?'] = False
    ages_df['intermediate intrusive?'] = False
    ages_df['Exclude?'] = False
    ages_df['distribution_type'] = 'Normal'

    sample_df['Exclude?'] = False

    return ages_df, sample_df

def synthetic_observations_from_prior(age_vector, ages_df, sample_heights = None,  uniform_heights = False, samples_per_section = 20, proxies = ['d13c'], proxy_std = 0.1, seed = None, ls_dist = 'Wald', ls_min = 0, ls_mu = 20, ls_lambda = 50, ls_sigma = 50, var_sigma = 10, white_noise_sigma = 1e-1,  gp_mean_mu = 0, gp_mean_sigma = 10, approximate = False, hsgp_m = 15, hsgp_c = 1.3, offset_type = 'section', offset_prior = 'Laplace', offset_alpha = 0, offset_beta = 1, offset_sigma = 1, offset_mu = 0, offset_b = 2, noise_type = 'section', noise_prior = 'HalfCauchy', noise_beta = 1, noise_sigma = 1, noise_nu = 1, jitter = 0.001, **kwargs):

    """
    Given age constraints for a set of stratigraphic sections in ``ages_df``, generate synthetic proxy observations by sampling the model prior. Accepts all arguments that can be passed to :py:meth:`build_model() <stratmc.model.build_model>` in :py:mod:`stratmc.model`.

    Parameters
    ----------
    age_vector: np.array(float)
        Vector of ages at which to evaluate synthetic proxy signal(s).

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for synthetic sections.

    sample_heights: dict{list(float) or numpy.array(float)}, optional
        Sample heights for each stratigraphic section in ``ages_df``; must be a dictionary with section names as keys. Defaults to ``None``, which results in either uniformly spaced or randomly spaced sample heights (depending on the ``uniform_heights`` argument).

    uniform_heights: bool, optional
        Whether to generate uniformly spaced (set to ``True``) or randomly spaced (set to ``False``) sample heights if dictionary of ``sample_heights`` not provided. Defaults to ``False`` (randomly spaced samples).

    samples_per_section: int or dict(int), optional
        Number of samples per section to generate if ``sample_heights`` not provided; either an integer (if the same for all sections) or a dictionary with section names as keys. Defaults to 20.

    proxies: list(str), optional
        List of proxies to generate synthetic observations for. Defaults to `d13c`.

    proxy_std: float or dict(float), optional
        Measurement uncertainty for each proxy; pass a dictionary of floats with the elements of ``proxies`` as keys to use a different value for each proxy, or an integer to use the same value for all proxies. Defaults to 0.1.

    seed: int, optional
        Seed to use while generating synthetic observations.

    Returns
    -------
    signals: dict(float)
        Tracers signals drawn from the model prior (evaluated at the points in ``age_vector``) used to generate synthetic observations; dictionary keys are ``proxies``.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for synthetic stratigraphic sections.

    prior: arviz.InferenceData
        An  :class:`arviz.InferenceData` object containing the prior draw from the model used to generate synthetic observations.

    model: pymc.Model
        :class:`pymc.model.core.Model` object used to generate synthetic observations.

    """

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])

    else:
        sections = list(np.unique(ages_df['section']))

    np.random.seed(seed)

    if (len(proxies) == 1) & (type(proxy_std) != dict):
        std = proxy_std
        proxy_std = {}
        proxy_std[proxies[0]] = std

    elif (len(proxies) != 1) & (type(proxy_std) != dict):
        std = proxy_std
        proxy_std = {}
        for proxy in proxies:
            proxy_std[proxy] = std

    # if sample heights are not provided, then generate synthetic 'sample heights' with np.random.uniform in between the minimum and maximum age heights
    if sample_heights is None:
        if type(samples_per_section) == int:
            n = samples_per_section
            samples_per_section = {}
            for section in sections:
                samples_per_section[section] = n

        sample_heights = {}
        for section in sections:
            min_height = np.min(ages_df[ages_df['section']==section]['height'].values)
            max_height = np.max(ages_df[ages_df['section']==section]['height'].values)

            if uniform_heights:
                sample_heights[section] = np.linspace(min_height + 1, max_height - 1, samples_per_section[section])

            else:
                sample_heights[section] = np.sort(np.random.uniform(min_height + 0.01, max_height - 0.01, size = samples_per_section[section]))

    # if sample heights not provided and samples per section not specified
    elif type(sample_heights) != dict:
            sys.exit(f"sample_heights must be a dictionary (keys = section names, values = array or list of sample heights)")

    sample_df_columns = ['section', 'height', 'Exclude?'] + [proxy for proxy in proxies]
    sample_df = pd.DataFrame(columns = sample_df_columns)

    for section in sections:
        section_dict = {
        'section': [section] * len(sample_heights[section]),
        'height': sample_heights[section],
        'Exclude?': [False] * len(sample_heights[section])
        }

        for proxy in proxies:
            section_dict[proxy] = [0] * len(sample_heights[section])
            section_dict[proxy + '_std'] = np.ones(len(sample_heights[section])) * proxy_std[proxy]

        sample_df = pd.concat([sample_df, pd.DataFrame.from_dict(section_dict)], ignore_index = True)

    for proxy in proxies:
        sample_df[proxy] = sample_df[proxy].astype(float)
        sample_df[proxy + '_std'] = sample_df[proxy + '_std'].astype(float)

    # make sure ages_df has all of the required columns
    ages_df_columns = ['distribution_type', 'param_1', 'param_2', 'param_1_name', 'param_2_name', 'shared?', 'name', 'Exclude?', 'intermediate detrital?', 'intermediate intrusive?']

    for col in ages_df_columns:
        if col not in list(ages_df):
            ages_df[col] = np.nan

    ages_df['shared?'] = False
    ages_df['intermediate detrital?'] = False
    ages_df['intermediate intrusive?'] = False
    ages_df['Exclude?'] = False
    ages_df['distribution_type'] = 'Normal'

    sample_df.sort_values(by = ['section', 'height'], inplace = True)
    sections = np.unique(sample_df['section'])

    sample_df['Exclude?'] = sample_df['Exclude?'].astype(bool)

   # build a model using the synthetic data (set proxy_observed = False in build_model)
    model, gp = build_model(sample_df,
                            ages_df,
                            sections = sections,
                            proxies = proxies,
                            proxy_sigma_default = proxy_std,
                            ls_dist = ls_dist,
                            ls_min = ls_min,
                            ls_mu = ls_mu,
                            ls_lambda = ls_lambda,
                            ls_sigma = ls_sigma,
                            var_sigma = var_sigma,
                            white_noise_sigma = white_noise_sigma,
                            gp_mean_mu = gp_mean_mu,
                            gp_mean_sigma = gp_mean_sigma,
                            approximate = approximate,
                            offset_type = offset_type,
                            offset_prior = offset_prior,
                            offset_alpha = offset_alpha,
                            offset_beta = offset_beta,
                            offset_sigma = offset_sigma,
                            offset_mu = offset_mu,
                            offset_b = offset_b,
                            noise_type = noise_type,
                            noise_prior = noise_prior,
                            noise_beta = noise_beta,
                            noise_sigma = noise_sigma,
                            noise_nu = noise_nu,
                            hsgp_m = hsgp_m,
                            hsgp_c = hsgp_c,
                            jitter = jitter,
                            proxy_observed = False
                            )

    with model:
        # single draw from the prior
        for proxy in proxies:
            f_pred = gp[proxy].conditional('f_pred_' + proxy, Xnew = age_vector, jitter = jitter)

        prior = pm.sample_prior_predictive(draws = 1, random_seed = seed)

    signals = {}
    for proxy in proxies:
        signals[proxy] = az.extract(prior.prior)['f_pred_' + proxy].values

        sample_df[proxy] = az.extract(prior.prior)[proxy + '_pred'].values

    sample_df['age'] = 0.0
    for section in sections:
        sample_df.loc[sample_df['section']==section, 'age'] = az.extract(prior.prior)[section + '_ages'].values.ravel()

    return signals, sample_df, prior, model


def synthetic_signal_from_prior(ages, num_signals = 100, ls_dist = 'Wald', ls_min = 0, ls_mu = 20, ls_lambda = 50, ls_sigma = 50, var_sigma = 10, gp_mean_mu = 0, gp_mean_sigma = 5, seed = None):

    """
    Draws synthetic signals from the model prior, and returns the signal conditioned over the points in ``ages``. To generate both signals and synthetic stratigraphic sections, instead use :py:meth:`synthetic_observations_from_prior() <stratmc.synthetics>`.

    Parameters
    ----------
    ages: numpy.array(float)
        Array of ages over which to condition the signal.

    num_signals: int, optional
        Number of signals to draw from prior. Defaults to 100.

    ls_dist: str, optional
        Prior distribution for the lengthscale hyperparameter of the exponential quadratic covariance kernel (:class:`pymc.gp.cov.ExpQuad <pymc.gp.cov.ExpQuad>`); set to ``Wald`` (:class:`pymc.Wald`) or ``HalfNormal`` (:class:`pymc.HalfNormal`). Defaults to ``Wald`` with ``mu = 20`` and ``lambda = 50``; to change ``mu`` and ``lambda``, pass the ``ls_mu`` and ``ls_lambda`` parameters. For ``HalfNormal``, the variance defaults to ``sigma = 50``; change by passing ``ls_sigma``.

    ls_min: float, optional
        Minimum value for the lengthscale hyperparameter of the :class:`pymc.gp.cov.ExpQuad` covariance kernel; shifts the lengthscale prior by ``ls_min``. Defaults to 0.

    ls_mu: float, optional
        Mean (`mu`) of the :class:`pymc.gp.cov.ExpQuad` lengthscale prior if ``ls_dist = `Wald```. Defaults to 20.

    ls_lambda: float, optional
        Relative precision (`lam`) of the :class:`pymc.gp.cov.ExpQuad` lengthscale hyperparameter prior if ``ls_dist = `Wald```. Defaults to 50.

    ls_sigma: float, optional
        Scale parameter (`sigma`) of the :class:`pymc.gp.cov.ExpQuad` lengthscale hyperparameter prior if ``ls_dist = `HalfNormal```. Defaults to 50.

    var_sigma: float, optional
        Scale parameter (`sigma') of the covariance kernel variance hyperparameter prior, which is a :class:`pymc.HalfNormal` distribution. Defaults to 10.

    gp_mean_mu: float, optional
        Mean (`mu`) of the GP mean function prior, which is a :class:`pymc.Normal` distribution. Defaults to 0.

    gp_mean_sigma: float, optional
        Standard deviation (`sigma`) of the GP mean function prior, which is a :class:`pymc.Normal` distribution. Defaults to 5.

    seed: int, optional
        Random seed used to generate signals.

    Returns
    -------
    signal: numpy.ndarray(float)
        Array with shape ``ages x number of signals`` containing the ``n = num_signals`` synthetic signals drawn from the prior.
    """

    with pm.Model() as model:
        # proxy GP
        if ls_dist == 'Wald':
            gp_ls_unshifted = pm.Wald('gp_ls_unshifted', mu = ls_mu, lam = ls_lambda, shape = 1)

        elif ls_dist == 'HalfNormal':
            gp_ls_unshifted = pm.HalfNormal('gp_ls_unshifted', sigma = ls_sigma, shape = 1)

        gp_ls = pm.Deterministic('gp_ls', gp_ls_unshifted + ls_min)

        gp_var = pm.HalfNormal('gp_var', sigma = var_sigma, shape = 1)

        m_proxy = pm.Normal('m', mu = gp_mean_mu, sigma = gp_mean_sigma, shape = 1)

        # mean and covariance functions
        mean_fun = pm.gp.mean.Constant(m_proxy)
        cov1 = gp_var ** 2 * pm.gp.cov.ExpQuad(1, gp_ls)

        # GP prior
        gp = pm.gp.Latent(mean_func = mean_fun, cov_func = cov1)

        f = gp.prior('f', X=ages[:,None],
                         reparameterize=True)

        prior = pm.sample_prior_predictive(draws = num_signals, random_seed = seed)

    signals = az.extract(prior.prior)['f'].values

    return signals, prior


def quantify_signal_recovery(full_trace, true_signal, proxy = 'd13c', mode = 'posterior'):

    """
    Calculates the likelihood of the true signal (for synthetic tests, where the true signal is known) given draws from the posterior (default) or prior. The likelihood is evaluated at each age (the posterior signal and the true signal must be evaluated at the same ages). Provides a measure of signal recovery.

    Parameters
    ----------
    full_trace: arviz.InferenceData or list(arviz.InferenceData)
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`. If passed as a list, the posterior draws for all traces will be combined when calculating `posterior_likelihood`.

    true_signal: np.array
        True values for the proxy signal, evaluated at the same ages as the posterior signal in ``full_trace``.

    proxy: str, optional
        Tracer signal to evaluate. Defaults to `d13c'.

    mode: str, optional
        Whether to use the posterior or prior to calculate signal recovery. Defaults to `posterior`.

    Returns
    -------
    posterior_likelihood: np.array
        Array of posterior likelihoods (evaluated at each age).

    """

    if mode == 'posterior':
        if type(full_trace) is list:
            for i in np.arange(len(full_trace)):
                if i == 0:
                    post_proxy = az.extract(full_trace[i].posterior_predictive)['f_pred_' + proxy].values
                else:
                    post_proxy_temp = az.extract(full_trace[i].posterior_predictive)['f_pred_' + proxy].values
                    post_proxy = np.hstack([post_proxy, post_proxy_temp])
        else:
            post_proxy = az.extract(full_trace.posterior_predictive)['f_pred_' + proxy].values

    elif mode == 'prior':
        if type(full_trace) is list:
            for i in np.arange(len(full_trace)):
                if i == 0:
                    post_proxy = az.extract(full_trace[i].prior)['f_pred_' + proxy].values
                else:
                    post_proxy_temp = az.extract(full_trace[i].prior)['f_pred_' + proxy].values
                    post_proxy = np.hstack([post_proxy, post_proxy_temp])
        else:
            post_proxy = az.extract(full_trace.prior)['f_pred_' + proxy].values

    if len(true_signal) != post_proxy.shape[0]:
        sys.exit(f"Length of true_signal does not match length of posterior predictive signal. Check that both are evaluated at the same ages (if not, interpolate true_signal to the ages at which the posterior signal was evaluated).")

    posterior_likelihood = []
    for i in np.arange(post_proxy.shape[0]):
        X = post_proxy[i, :]
        kde = gaussian_kde(X)
        posterior_likelihood.append(kde.evaluate(true_signal[i])[0])

    return np.array(posterior_likelihood)

def sample_age_recovery(full_trace, sample_df, sections = None, mode = 'posterior'):

    """
    Calculates the likelihood of the true sample ages (for synthetic tests, where the true age of each sample is known) given draws from the posterior (default) or prior. Provides a measure of age model recovery.

    Parameters
    ----------
    full_trace: arviz.InferenceData or list(arviz.InferenceData)
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`. If passed as a list, the posterior draws for all traces will be combined when calculating `posterior_likelihood`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for synthetic sections.

    sections: list(str) or numpy.array(str), optional
        List of sections to evaluate. Defaults to all sections in sample_df.

    mode: str, optional
         Whether to use the posterior or prior age models. Defaults to `posterior`.

    Returns
    -------
    posterior_likelihood: dict{float} or np.array(float)
        Posterior likelihoods for the true age of each sample. Returned as an array if only one section is evaluated, or a dictionary of arrays if multiple sections are evaluated.

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

    if sections is None:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    if len(sections) > 1:
        posterior_likelihood = {}

    for section in sections:
        section_df = sample_df[sample_df['section'] == section]
        section_df.sort_values(by = 'height', inplace = True)
        true_ages = section_df['age'].values

        if mode == 'posterior':
            if type(full_trace) is list:
                for i in np.arange(len(full_trace)):
                    if i == 0:
                        post_ages = az.extract(full_trace[i].posterior)[str(section) + '_ages'].values
                    else:
                        post_ages_temp = az.extract(full_trace[i].posterior)[str(section) + '_ages'].values
                        post_ages = np.hstack([post_ages, post_ages_temp])
            else:
                post_ages = az.extract(full_trace.posterior)[str(section) + '_ages'].values

        elif mode == 'prior':
            if type(full_trace) is list:
                for i in np.arange(len(full_trace)):
                    if i == 0:
                        post_ages = az.extract(full_trace[i].prior)[str(section) + '_ages'].values
                    else:
                        post_ages_temp = az.extract(full_trace[i].prior)[str(section) + '_ages'].values
                        post_ages = np.hstack([post_ages, post_ages_temp])
            else:
                post_ages = az.extract(full_trace.prior)[str(section) + '_ages'].values

        if len(true_ages) != post_ages.shape[0]:
            sys.exit(f"Number of samples in sample_df does not match the number in the trace")


        if len(sections) > 1:
            posterior_likelihood[section] = []

        else:
            posterior_likelihood = []

        for i in np.arange(post_ages.shape[0]):
            X = post_ages[i, :]
            kde = gaussian_kde(X)

            if len(sections) > 1:
                posterior_likelihood[section].append(kde.evaluate(true_ages[i])[0])

            else:
                posterior_likelihood.append(kde.evaluate(true_ages[i])[0])


    return posterior_likelihood


def sample_age_residuals(full_trace, sample_df, sections = None, mode = 'posterior'):
    """
    Calculates the residual (for each draw) between the true age and the posterior (default) or prior age of each sample.

    Parameters
    ----------
    full_trace: arviz.InferenceData or list(arviz.InferenceData)
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`. If passed as a list, the posterior draws for all traces will be combined when calculating `age_residuals`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for synthetic sections.

    sections: list(str) or numpy.array(str), optional
        List of sections to evaluate. Defaults to all sections in sample_df.

    mode: str, optional
         Whether to use the posterior or prior age models. Defaults to `posterior`.

    Returns
    -------
    age_residuals: np.array or dict{np.array}
        Sample age residuals; shape is (number of samples, number of posterior draws). Returned as an array if only one section is evaluated, or a dictionary of arrays if multiple sections are evaluated.

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

    if sections is None:
            sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    if len(sections) > 1:
        age_residuals = {}

    for section in sections:
        true_ages = sample_df[sample_df['section'] == section]['age'].values

        if mode == 'posterior':
            if type(full_trace) is list:
                for i in np.arange(len(full_trace)):
                    if i == 0:
                        post_ages = az.extract(full_trace[i].posterior)[str(section) + '_ages'].values
                    else:
                        post_ages_temp = az.extract(full_trace[i].posterior)[str(section) + '_ages'].values
                        post_ages = np.hstack([post_ages, post_ages_temp])
            else:
                post_ages = az.extract(full_trace.posterior)[str(section) + '_ages'].values

        elif mode == 'prior':
            if type(full_trace) is list:
                for i in np.arange(len(full_trace)):
                    if i == 0:
                        post_ages = az.extract(full_trace[i].prior)[str(section) + '_ages'].values
                    else:
                        post_ages_temp = az.extract(full_trace[i].prior)[str(section) + '_ages'].values
                        post_ages = np.hstack([post_ages, post_ages_temp])
            else:
                post_ages = az.extract(full_trace.prior)[str(section) + '_ages'].values

        true_ages_array = np.repeat(true_ages, post_ages.shape[1]).reshape(post_ages.shape)

        if len(sections) > 1:
            age_residuals[section] = true_ages_array - post_ages

        else:
            age_residuals =  true_ages_array - post_ages

    return age_residuals
