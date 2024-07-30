import warnings

import numpy as np

from stratmc.config import PROJECT_ROOT
from stratmc.data import load_data
from stratmc.inference import get_trace
from stratmc.model import build_model

warnings.filterwarnings("ignore", ".*X_new group is not defined in the InferenceData scheme.*")


def test_custom_priors():
    # load data
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df', proxies = ['d13c', 'd18o', 'd34s'], proxy_sigma_default = {'d13c': 0.1, 'd18o': 0.25, 'd34s': 0.5})

    # for this test, remove the d18o_std column to make sure that it's filled in w/in build_model if necessary
    sample_df['d18o_std'] = np.nan

    # build model
    d13c_offset_params = {'param_1_name': 'mu',
                      'param_1': 0,
                      'param_2_name': 'b',
                      'param_2': 3

    }

    d18o_offset_params = {'param_1_name': 'mu',
                        'param_1': 0,
                        'param_2_name': 'sigma',
                        'param_2': 2

    }

    model, _ = build_model(
                        sample_df,
                        ages_df,
                        proxies = ['d13c', 'd18o'],
                        ls_dist = {'d13c': 'Wald', 'd18o': 'HalfNormal'},
                        ls_min = {'d13c': 1, 'd18o': 1.5}, # minimum RBF kernel lengthscale
                        ls_mu =  {'d13c': 5}, # mean of Wald distribution used as RBF kernel lengthscale prior
                        ls_sigma = {'d18o': 10},
                        ls_lambda = {'d13c': 15, 'd18o': 20}, # lambda of Wald distribution used as RBF kernel lengthscale prior
                        var_sigma = {'d13c': 5, 'd18o': 10},
                        offset_type = 'groups', # custom offset  groups for both proxies
                        noise_type = {'d13c': 'groups','d18o': 'section'},
                        offset_prior = {'d13c': 'Laplace', 'd18o': 'Normal'},
                        offset_params = {'d13c': d13c_offset_params, 'd18o': d18o_offset_params},
                        noise_prior = {'d13c': 'HalfNormal', 'd18o': 'HalfStudentT'}
)

    assert 'shallow_group_offset_d13c' in str(np.array(model.basic_RVs))
    assert 'deep_group_offset_d13c' in str(np.array(model.basic_RVs))
    assert 'shallow_group_offset_d18o' in str(np.array(model.basic_RVs))
    assert 'deep_group_offset_d18o' in str(np.array(model.basic_RVs))
    assert 'shallow_group_noise_d13c' in str(np.array(model.basic_RVs))
    assert 'deep_group_noise_d13c' in str(np.array(model.basic_RVs))
    assert '1_section_noise_d18o' in str(np.array(model.basic_RVs))


def test_sample_numpyro():
    # load data
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df', proxies = ['d13c', 'd18o', 'd34s'], proxy_sigma_default = {'d13c': 0.1, 'd18o': 0.25, 'd34s': 0.5})

    age_min = np.min(ages_df['age'])
    age_max = np.max(ages_df['age'])

    predict_ages = np.arange(age_min, age_max + 0.5, 0.5)[:,None]

    model, gp = build_model(
                        sample_df,
                        ages_df,
                        proxies = ['d13c', 'd18o'],
                        ls_dist = 'Wald',
                        ls_min = 1, # minimum RBF kernel lengthscale
                        ls_mu = 5, # mean of Wald distribution used as RBF kernel lengthscale prior
                        ls_lambda = 15, # lambda of Wald distribution used as RBF kernel lengthscale prior
                        offset_type = 'section', # per-section offset with default prior
                        noise_type = 'section', # per-section noise with default prior
                        )

    _ = get_trace(model,
                        gp,
                        predict_ages,
                        sample_df,
                        ages_df,
                        proxies = ['d13c', 'd18o'],
                        chains = 2,
                            tune = 2,
                            draws = 2,
                            prior_draws = 2,
                        target_accept = 0.9,
                        save = False,
                        sampler = 'numpyro'
                        )

def test_sample_blackjax():
    # load data
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df', proxies = ['d13c', 'd18o', 'd34s'], proxy_sigma_default = {'d13c': 0.1, 'd18o': 0.25, 'd34s': 0.5})

    age_min = np.min(ages_df['age'])
    age_max = np.max(ages_df['age'])

    predict_ages = np.arange(age_min, age_max + 0.5, 0.5)[:,None]

    model, gp = build_model(
                        sample_df,
                        ages_df,
                        proxies = ['d13c', 'd18o'],
                        ls_dist = 'Wald',
                        ls_min = 1, # minimum RBF kernel lengthscale
                        ls_mu = 5, # mean of Wald distribution used as RBF kernel lengthscale prior
                        ls_lambda = 15, # lambda of Wald distribution used as RBF kernel lengthscale prior
                        offset_type = 'section', # per-section offset with default prior
                        noise_type = 'section', # per-section noise with default prior
                        )

    _ = get_trace(model,
                        gp,
                        predict_ages,
                        sample_df,
                        ages_df,
                        proxies = ['d13c', 'd18o'],
                        chains = 2,
                            tune = 2,
                            draws = 2,
                            prior_draws = 2,
                        target_accept = 0.9,
                        save = False,
                        sampler = 'blackjax'
                        )

def test_hsgp():
    # load data
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df', proxies = ['d13c', 'd18o', 'd34s'], proxy_sigma_default = {'d13c': 0.1, 'd18o': 0.25, 'd34s': 0.5})

    age_min = np.min(ages_df['age'])
    age_max = np.max(ages_df['age'])

    predict_ages = np.arange(age_min, age_max + 0.5, 0.5)[:,None]

    model, gp = build_model(
                        sample_df,
                        ages_df,
                        approximate = True,
                        hsgp_m = 5,
                        hsgp_c = 1.3,
                        proxies = ['d13c', 'd18o'],
                        ls_dist = 'Wald',
                        ls_min = 1, # minimum RBF kernel lengthscale
                        ls_mu = 5, # mean of Wald distribution used as RBF kernel lengthscale prior
                        ls_lambda = 15, # lambda of Wald distribution used as RBF kernel lengthscale prior
                        offset_type = 'section', # per-section offset with default prior
                        noise_type = 'section', # per-section noise with default prior
                        )

    _ = get_trace(model,
                        gp,
                        predict_ages,
                        sample_df,
                        ages_df,
                        proxies = ['d13c', 'd18o'],
                        chains = 2,
                            tune = 2,
                            draws = 2,
                            prior_draws = 2,
                        target_accept = 0.9,
                        save = False,
                        sampler = 'blackjax',
                        approximate = True
                        )
