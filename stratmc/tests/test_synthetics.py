import numpy as np
import pandas as pd

from stratmc.data import load_object, load_trace
from stratmc.synthetics import (
    make_excursion,
    quantify_signal_recovery,
    sample_age_recovery,
    sample_age_residuals,
    synthetic_observations_from_prior,
    synthetic_sections,
    synthetic_signal_from_prior,
)


def test_make_excursion():
    age_vector = np.arange(400, 450, 0.25)
    amplitude = [3, -4, 6, -2]

    d13c_signal = make_excursion(
        age_vector,
        amplitude,
        rising_time=[0.3, 0.7, 0.9, 0.3],
        smooth=True,
        smoothing_factor=2,
        rate_offset=True,
        seed=7,
    )

    assert len(d13c_signal) == len(age_vector)


def test_synthetic_sections():
    proxies = ['d13c', 'd34s']
    age_vector = np.arange(400, 450, 0.25)
    amplitude = [3, -4, 6, -2]

    d13c_signal = make_excursion(
        age_vector,
        amplitude,
        excursion_duration=[10, 8, 10, 5],
        rising_time=[0.3, 0.7, 0.9, 0.3],
        smooth=True,
        smoothing_factor=2,
        rate_offset=True,
        seed=7,
    )

    amplitude = [15, 30, 10, 25]
    d34s_signal = make_excursion(
        age_vector,
        amplitude,
        baseline=20,
        excursion_duration=[10, 8, 10, 5],
        smooth=False,
        smoothing_factor=2,
        rate_offset=False,
        seed=7,
    )

    signal_dict = {}
    signal_dict["d13c"] = d13c_signal
    signal_dict["d34s"] = d34s_signal

    num_samples =  30
    num_sections = 4
    section_thickness = 30
    section_seed = 25
    section_noise = False
    section_noise_amp = {}
    section_noise_amp['d13c'] = 0.75
    section_noise_amp['d34s'] = 2


    sections = []
    age_constraints = {}
    age_constraints_std = {}
    for i in np.arange(num_sections):
        sections.append(str(i))
        np.random.seed(0)

        age_constraints[str(i)] = np.array([400, 450])
        age_constraints_std[str(i)] = np.random.uniform(0.5, 1.5, size = len(age_constraints[str(i)]))

    ages_df, sample_df = synthetic_sections(age_vector, signal_dict, num_sections, num_samples, section_thickness, noise = section_noise, noise_amp = section_noise_amp, seed = section_seed, age_constraints = age_constraints, age_constraints_std = age_constraints_std, proxies = proxies)


def test_synthetic_observations_from_prior():

    # random ages for 6 input sections
    age_vector = np.linspace(200, 500, 300)[:, None]

    sections = ['1', '2', '3']

    section_ages = {}
    section_age_std = {}
    section_age_heights = {}

    seeds = {}
    seeds['1'] = 1
    seeds['2'] = 210
    seeds['3'] = 302


    ages_df = pd.DataFrame(columns = ['section', 'age', 'age_std', 'height'])
    for section in sections:
        np.random.seed(seeds[section])
        section_ages[section] = np.concatenate([np.random.normal(np.random.uniform(175, 300, 1), 30, 1), np.random.normal(np.random.uniform(400, 525, 1), 15, 1)])
        section_age_std[section] = np.random.uniform(2, 10, size = 2)
        section_age_heights[section] = np.array([50, 0])

        section_dict = {
        'section': [section] * 2,
        'age': section_ages[section],
        'age_std': section_age_std[section],
        'height': section_age_heights[section]
        }

        ages_df = pd.concat([ages_df, pd.DataFrame.from_dict(section_dict)], ignore_index = True)

    signals, sample_df, prior, model = synthetic_observations_from_prior(age_vector,
                                                               ages_df,
                                                               sections = sections,
                                                               samples_per_section = 25,
                                                               ls_dist = 'Wald',
                                                               ls_mu = 50,
                                                               seed = 302,
                                                              noise_prior = 'HalfNormal',
                                                              noise_beta = 1,
                                                              noise_sigma = 1,
                                                              uniform_heights = True,
                                                              proxies = ['d13c', 'd18o']
                                                                )

    assert len(signals['d13c']) == len(age_vector)
    assert len(signals['d18o']) == len(age_vector)

def test_synthetic_signal_from_prior():
    ages = np.arange(100, 125 + 0.5, 0.5)[:, None]

    num_signals = 100

    signals, _ = synthetic_signal_from_prior(ages.ravel(), num_signals = num_signals, ls_dist = 'Wald', ls_min = 0, ls_mu = 20, ls_lambda = 50, ls_sigma = 50, var_sigma = 10, gp_mean_mu = 0, gp_mean_sigma = 5, seed = None)

    assert signals.shape == (len(ages), num_signals)


# def test_quantify_signal_recovery():

# def test_sample_age_recovery():

# def test_sample_age_residuals():
