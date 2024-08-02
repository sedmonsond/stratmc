import warnings
from unittest.mock import patch

from stratmc.config import PROJECT_ROOT
from stratmc.data import load_data, load_trace
from stratmc.inference import extend_age_model, interpolate_proxy
from stratmc.plotting import (
    accumulation_rate_stratigraphy,
    age_constraints,
    age_height_model,
    covariance_hyperparameters,
    interpolated_proxy_inference,
    lengthscale_stability,
    lengthscale_traceplot,
    limiting_age_constraints,
    noise_summary,
    offset_summary,
    proxy_data_density,
    proxy_data_gaps,
    proxy_inference,
    proxy_signal_stability,
    proxy_strat,
    sadler_plot,
    sample_ages,
    sample_ages_per_chain,
    section_age_range,
    section_proxy_residuals,
    section_proxy_signal,
    section_summary,
)

warnings.filterwarnings("ignore", ".*The group X_new is not defined in the InferenceData scheme.*")
warnings.filterwarnings("ignore", ".*X_new group is not defined in the InferenceData scheme.*")

@patch("matplotlib.pyplot.show")
def test_proxy_strat(a):
    # load data
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')

    _ = proxy_strat(sample_df, ages_df, proxy = 'd13c', plot_excluded_samples = True)

@patch("matplotlib.pyplot.show")
def test_proxy_inference(a):
    # load data
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')

    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = proxy_inference(sample_df, ages_df, full_trace, plot_constraints = True, plot_data = True, plot_excluded_samples = True, plot_mean = True, plot_mle = True)
    _ = proxy_inference(sample_df, ages_df, full_trace, sections = ['1', '2'], plot_constraints = True, plot_data = True, section_legend = True, plot_excluded_samples = True, plot_mean = True, plot_mle = True, proxy = 'd18o', orientation = 'vertical', fontsize = 10, figsize = (4, 7))



@patch("matplotlib.pyplot.show")
def test_interpolated_proxy_inference(a):
    # this will also test the interpolation functions
    # load data
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')

    # load trace
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')
    predict_ages = full_trace.X_new.X_new.values

    interpolated_df = extend_age_model(full_trace, sample_df, ages_df, new_proxies = 'd34s')
    interpolated_proxy_df = interpolate_proxy(interpolated_df, 'd34s', predict_ages)
    _ = interpolated_proxy_inference(interpolated_df, interpolated_proxy_df, 'd34s', plot_data = True)

    assert interpolated_proxy_df.shape[0] == len(predict_ages)

@patch("matplotlib.pyplot.show")
def test_age_height_model(a):
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')

    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = age_height_model(sample_df, ages_df, full_trace)

@patch("matplotlib.pyplot.show")
def test_section_proxy_signal(a):
    # also tests map_ages_to_section
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = section_proxy_signal(full_trace, sample_df, ages_df, include_radiometric_ages = True, plot_constraints = True)
    _ = section_proxy_signal(full_trace, sample_df, ages_df, include_radiometric_ages = False, plot_constraints = True, yax = 'age')


@patch("matplotlib.pyplot.show")
def test_covariance_hyperparameters(a):
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = covariance_hyperparameters(full_trace)
    _ = covariance_hyperparameters(full_trace, proxy = 'd18o', fontsize = 10)

@patch("matplotlib.pyplot.show")
def test_section_summary(a):
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = section_summary(sample_df, ages_df, full_trace, '0', plot_excluded_samples = True, plot_noise_prior = True, plot_offset_prior = True)

    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_custom_priors')
    _ = section_summary(sample_df, ages_df, full_trace, '0', plot_excluded_samples = True, plot_noise_prior = True, plot_offset_prior = True)

@patch("matplotlib.pyplot.show")
def test_noise_summary(a):
    # d13C has group noise, d18O has per-section noise
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_custom_priors')
    _ = noise_summary(full_trace, fontsize = 10)
    _ = noise_summary(full_trace, proxy  = 'd18o')

@patch("matplotlib.pyplot.show")
def test_offset_summary(a):
    # trace w/ per-section offsets (for both d13c and d18o)
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = offset_summary(full_trace, fontsize = 10)

    # trace w/ group offsets (for both d13c and d18o)
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_custom_priors')

    _ = offset_summary(full_trace, proxy = 'd18o')

@patch("matplotlib.pyplot.show")
def test_section_proxy_residuals(a):
    sample_df, _ = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = section_proxy_residuals(full_trace, sample_df)

    _ = section_proxy_residuals(full_trace, sample_df, include_excluded_samples = True, proxy = 'd18o')

@patch("matplotlib.pyplot.show")
def test_sample_ages(a):
    sample_df, _ = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = sample_ages(full_trace, sample_df, '0', plot_excluded_samples = True)
    _ = sample_ages(full_trace, sample_df, '0', plot_excluded_samples = False)

@patch("matplotlib.pyplot.show")
def test_sample_ages_per_chain(a):
    sample_df, _ = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = sample_ages_per_chain(full_trace, sample_df, '0', plot_excluded_samples = True)
    _ = sample_ages_per_chain(full_trace, sample_df, '0', plot_excluded_samples = False, plot_prior = True)

@patch("matplotlib.pyplot.show")
def test_age_constraints(a):
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = age_constraints(full_trace, '1')

@patch("matplotlib.pyplot.show")
def test_limiting_age_constraints(a):
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = limiting_age_constraints(full_trace, sample_df, ages_df, '2')

@patch("matplotlib.pyplot.show")
def test_sadler_plot(a):
    # test with and without age constraints
    # this also tests accumulation_rate with method = 'all'
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = sadler_plot(full_trace, sample_df, ages_df, include_age_constraints = False, scale = 'linear')
    _ = sadler_plot(full_trace, sample_df, ages_df, include_age_constraints = True)
    _ = sadler_plot(full_trace, sample_df, ages_df, include_age_constraints = False, method = 'scatter')
    _ = sadler_plot(full_trace, sample_df, ages_df, include_age_constraints = False, scale = 'linear', method = 'scatter')

@patch("matplotlib.pyplot.show")
def test_accumulation_rate_stratigraphy(a):
    # test with and without age constraints
    # this also test accumulation_rate with method = 'successive'
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = accumulation_rate_stratigraphy(full_trace, sample_df, ages_df, include_age_constraints = True)
    _ = accumulation_rate_stratigraphy(full_trace, sample_df, ages_df, include_age_constraints = False, rate_scale = 'linear')

@patch("matplotlib.pyplot.show")
def test_section_age_range(a):
    # also tests age_range_to_height function
    sample_df, ages_df = load_data(str(PROJECT_ROOT) + '/examples/test_sample_df', str(PROJECT_ROOT) + '/examples/test_ages_df')
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = section_age_range(full_trace, sample_df, ages_df, 125, 130, legend = True)

@patch("matplotlib.pyplot.show")
def test_proxy_data_gaps(a):
    # this also tests find_gaps in inference module
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = proxy_data_gaps(full_trace)

    time_grid = np.arange(100, 150, 2)
    _ = proxy_data_gaps(full_trace, yaxis = 'count', time_grid = time_grid)

@patch("matplotlib.pyplot.show")
def test_proxy_data_density(a):
    # this also tests count_samples in inference module
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = proxy_data_density(full_trace)

    time_grid = np.arange(100, 150, 2)
    _ = proxy_data_density(full_trace, time_grid = time_grid)

@patch("matplotlib.pyplot.show")
def test_lengthscale_traceplot(a):
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = lengthscale_traceplot(full_trace)
    _ = lengthscale_traceplot(full_trace, chains = [1])

@patch("matplotlib.pyplot.show")
def test_lengthscale_stability(a):
    # this also tests calculate_lengthscale_stability in inference module
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = lengthscale_stability(full_trace)

@patch("matplotlib.pyplot.show")
def test_proxy_signal_stability(a):
    # this also tests calculate_proxy_signal_stability in inference module
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    _ = proxy_signal_stability(full_trace)
