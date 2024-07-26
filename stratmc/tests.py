import arviz as az
import numpy as np 
import pymc as pm


from stratmc.data import clean_data
from stratmc.model import build_model

            
def check_superposition(full_trace, sample_df, ages_df, sections): 
    """
    Test that stratigraphic superposition between all age constriants and samples is respected in the posterior.

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.
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

    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)
    
    ages_df = ages_df[(~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?'])]
    
    for section in sections:
        sample_heights = sample_df[sample_df['section']==section]['height']
        age_heights = ages_df[ages_df['section']==section]['height']

        comb = np.concatenate([sample_heights, age_heights])
        sort_idx = np.argsort(comb)

        sample_posterior = az.extract(full_trace.posterior)[str(section) + '_ages'].values
        age_posterior = az.extract(full_trace.posterior)[str(section) + '_radiometric_age'].values

        posterior_stacked = np.vstack([sample_posterior, age_posterior])

        draws = posterior_stacked.shape[1]

        for i in range(draws):
            assert(all(np.diff(posterior_stacked[sort_idx,i].ravel()) <= 0)), "stratigraphic superposition violated in section " + str(section) + " draw " + str(i)
            
            
def check_section_superposition(full_trace, superposition_dict):
    """
    Check that superposition between sections included in ``superposition_dict`` passed to :py:meth:`build_model() <stratmc.model.build_model()>` in :py:mod:`stratmc.model` has been enforced in the posterior. 

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.
    
    superposition_dict: dict
        Superposition dictionary passed to passed to :py:meth:`build_model() <stratmc.model.build_model()>` in :py:mod:`stratmc.model`. Keys are section names, and values are lists of sections that must be older.
        
    """
    for section in list(superposition_dict.keys()): 
        section_ages = az.extract(full_trace.posterior)[str(section) + '_ages'].values

        older_section_ages = {}
        for older_section in superposition_dict[section]:
            older_section_ages = az.extract(full_trace.posterior)[str(older_section) + '_ages'].values
            assert np.min(older_section_ages.ravel()) > np.min(section_ages.ravel())
            
            
            
def check_detrital_ages(trace, sample_df, ages_df, sections = None):
    """
    Check that detrital age constraints have been enforced in the posterior. 

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.
    
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.
        
    """
        
    if sections is None: 
        sections = list(np.unique(sample_df['section']))
        
    for section in sections:
        section_df = sample_df[sample_df['section']==section]
        section_df.sort_values(by = 'height', inplace = True)
        
        section_age_df = ages_df[(ages_df['section']==section) & (~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?'])]
        intermediate_detrital_section_ages_df = ages_df[(ages_df['section']==section) & (ages_df['intermediate detrital?'])]

        heights = section_df['height'].values
        age_heights = section_age_df['height'].values
            
        section_sample_ages = az.extract(trace.posterior)[section + '_ages'].values

        for interval in np.arange(0, len(age_heights)-1).tolist():
            
            above = intermediate_detrital_section_ages_df['height']>age_heights[interval]
            below = intermediate_detrital_section_ages_df['height']<age_heights[interval+1]
            detrital_interval_df = intermediate_detrital_section_ages_df[above & below]
        
        
            for height, shared, name, i in zip (detrital_interval_df['height'], detrital_interval_df['shared'], detrital_interval_df['name'], np.arange(detrital_interval_df['height'].shape[0])):
                sample_idx = section_df['height'] > height

                # check that all the posterior ages for overlying samples are younger than the detrital age during each draw
                if shared: 
                    detrital_age_posterior = az.extract(trace.posterior)[name].values
                else: 
                    dist_name = str(section)+'_'+ str(interval) +'_' + 'detrital_age_' + str(i)
                    detrital_age_posterior = az.extract(trace.posterior)[dist_name].values

                for draw in np.arange(section_sample_ages.shape[1]):
                    assert all(section_sample_ages[:, draw][sample_idx] <= detrital_age_posterior[draw])
                
    
def check_intrusive_ages(trace, sample_df, ages_df, sections = None):
    """
    Check that intrusive age constraints have been enforced in the posterior. 

    Parameters
    ----------

    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.
    
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.
        
    """
    if sections is None: 
        sections = list(np.unique(sample_df['section']))
        
    for section in sections:
        section_df = sample_df[sample_df['section']==section]
        section_df.sort_values(by = 'height', inplace = True)
        
        section_age_df = ages_df[(ages_df['section']==section) & (~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?'])]
        intermediate_intrusive_section_ages_df = ages_df[(ages_df['section']==section) & (ages_df['intermediate intrusive?'])]

        heights = section_df['height'].values
        age_heights = section_age_df['height'].values
            
        section_sample_ages = az.extract(trace.posterior)[section + '_ages'].values

        for interval in np.arange(0, len(age_heights)-1).tolist():
            
            above = intermediate_intrusive_section_ages_df['height']>age_heights[interval]
            below = intermediate_intrusive_section_ages_df['height']<age_heights[interval+1]
            intrusive_interval_df = intermediate_intrusive_section_ages_df[above & below]
        
        
            for height, shared, name, i in zip (intrusive_interval_df['height'], intrusive_interval_df['shared'], intrusive_interval_df['name'], np.arange(intrusive_interval_df['height'].shape[0])):
                sample_idx = section_df['height'] < height

                # check that all the posterior ages for overlying samples are younger than the detrital age during each draw
                if shared: 
                    intrusive_age_posterior = az.extract(trace.posterior)[name].values
                else: 
                    dist_name = str(section)+'_'+ str(interval) +'_' + 'intrusive_age_' + str(i)
                    intrusive_age_posterior = az.extract(trace.posterior)[dist_name].values

                for draw in np.arange(section_sample_ages.shape[1]):
                    assert all(section_sample_ages[:, draw][sample_idx] >= intrusive_age_posterior[draw])
