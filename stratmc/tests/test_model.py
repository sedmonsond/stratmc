from bayestrat.config import PROJECT_ROOT
from bayestrat.data import load_data
from bayestrat.model import build_model
from bayestrat.inference import get_trace

import numpy as np

def test_sample_numpyro():
    # test detrital, intermediate, and custom age distribution functionality in here 
    sample_df, ages_df = load_data('benchmark_sample_data', 'benchmark_ages')

    ages_new = np.linspace(np.max(ages_df['age']), np.min(ages_df['age']), 10)[:,None]

    model, gp = build_model(sample_df, ages_df, tracers = ['d13c'])
    full_trace = get_trace(model, 
                           gp, 
                           ages_new, 
                           tune = 1, 
                           draws=1, 
                           sampler = 'numpyro')

# def test_sample_blackjax():

# def test_hsgp():

# def test_multiproxy():
    
        
        
    
    
    
