import numpy as np
import pandas as pd

from stratmc.config import PROJECT_ROOT
from stratmc.data import (
    combine_data,
    combine_duplicates,
    combine_traces,
    drop_chains,
    load_object,
    load_trace,
    thin_trace,
)


def test_combine_data():
    sample_df = load_object(str(PROJECT_ROOT) +  '/examples/test_sample_df')
    sample_df_d13c = load_object(str(PROJECT_ROOT) +  '/examples/test_sample_df_d13c_only')
    sample_df_d18o = load_object(str(PROJECT_ROOT) +  '/examples/test_sample_df_d18o_only')
    sample_df_d34s = load_object(str(PROJECT_ROOT) +  '/examples/test_sample_df_d34s_only')

    comb_sample_df = combine_data([sample_df_d13c, sample_df_d18o, sample_df_d34s])

    assert comb_sample_df.shape == sample_df.shape

def test_combine_duplicates():
    sample_df = load_object(str(PROJECT_ROOT) + '/examples/test_sample_df')

    np.random.seed(0)
    sample_df_altered = sample_df.copy()
    sample_df_altered.loc[~np.isnan(sample_df_altered['d13c']), 'd13c'] += np.random.normal(0, 1, len(sample_df_altered['d13c'][~np.isnan(sample_df_altered['d13c'])].values))
    sample_df_altered.loc[~np.isnan(sample_df_altered['d18o']), 'd18o'] += np.random.normal(0, 1, len(sample_df_altered['d18o'][~np.isnan(sample_df_altered['d18o'])].values))
    sample_df_altered.loc[~np.isnan(sample_df_altered['d34s']), 'd34s'] += np.random.normal(0, 1, len(sample_df_altered['d34s'][~np.isnan(sample_df_altered['d34s'])].values))

    sample_df_stacked = pd.concat([sample_df, sample_df_altered], ignore_index = True)

    sample_df_combined = combine_duplicates(sample_df_stacked, proxies = ['d13c', 'd18o', 'd34s'])

    assert round(sample_df_combined.loc[0, 'd13c_population_std'], 6) == 0.882026
    assert round(sample_df_combined.loc[9, 'd18o_population_std'], 6) == 0.432218
    assert round(sample_df_combined.loc[14, 'd34s_population_std'], 6) == 0.524276


def test_combine_traces():
    full_trace_path = str(PROJECT_ROOT) + '/examples/traces/test_trace_1'

    full_trace_comb = combine_traces([full_trace_path, full_trace_path])

    assert len(list(full_trace_comb.posterior.chain.values)) == 4

def test_drop_chains():
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    dropped_trace = drop_chains(full_trace, [0])

    assert len(dropped_trace.posterior.chain.values) == 1

def test_thin_trace():
    full_trace = load_trace(str(PROJECT_ROOT) + '/examples/traces/test_trace_1')

    thinned_trace = thin_trace(full_trace, drop_freq = 2)

    assert len(thinned_trace.posterior.draw.values) == 5
