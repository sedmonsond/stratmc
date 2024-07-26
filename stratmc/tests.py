import arviz as az
import numpy as np 
import pymc as pm


from stratmc.data import clean_data
from stratmc.model import build_model
            
            
def check_superposition(full_trace, sample_df, ages_df, sections, **kwargs): 
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

def test_section_inference(sections, sample_df, ages_df, proxy = ['d13c'], draws = 500, tune = 500, target_accept = 0.8, seed=None):

    """

    .. todo:: 
        Maybe remove? 
        
    .. todo::
        Update approach for generating synthetic observations -- toy_data is deprecated
        
    Creates a new dataset for a list of input sections by drawing proxy observations from the model prior. Feeds that new data into the model and checks that the synthetic proxy signal is recovered within the 2-sigma bounds of the posterior.

    Parameters
    ----------
    section: list or numpy.array of str
        Sections to use for position data and age constraints.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    """

    toy_samples, toy_ages = toy_data(sections, sample_df, ages_df, random_seed=seed)

    model, gp = build_model(toy_samples, toy_ages, proxy = proxy)

    with model:
        prior = pm.sample_prior_predictive(samples=1, random_seed = seed)

    # save the true function sampled from the prior
    true_d13c = prior.prior['f'].values.ravel()
    true_age = prior.prior['ages'].values.ravel()

    new_samples = synthetic_data_from_prior(toy_samples, prior)

    new_model, new_gp = build_model(new_samples, ages_df, proxy = proxy)

    with new_model:
        posterior = pm.sampling.jax.sample_numpyro_nuts(draws = draws,
                                                        tune=tune,
                                                        target_accept=target_accept,
                                                        chains=4,
                                                        random_seed = seed)

        # since we only have 1 section, don't need f_pred (?)
        #fcond = gp.conditional('f_pred', Xnew=ages_new)
        v2r = ['f_pred']#[str(s) for s in model.deterministics]
        preds = pm.sample_posterior_predictive(posterior,
                                                      var_names=v2r,
                                                      return_inferencedata=True)

    # posterior samples from the underlying d13C function (at each input sample)
    mean_d13C_post = np.mean(preds.posterior_predictive['f'].values[0,:,:], axis = 0)
    std_d13C_post =  np.std(preds.posterior_predictive['f'].values[0,:,:], axis = 0)

    mean_age_post = np.mean(preds.posterior_predictive['ages'].values[0,:,:], axis = 0)
    std_age_post =  np.std(preds.posterior_predictive['ages'].values[0,:,:], axis = 0)

    # check that actual d13C is within 95%CI of predicted d13C
    d13c_in_bounds = all(
    (true_d13c < (np.percentile(preds.posterior_predictive['f'],97.5,axis=1).flatten()))
        & (true_d13c > (np.percentile(preds.posterior_predictive['f'],2.5,axis=1).flatten()))
    )

    # check that actual age is within 95% CI of predicted age
    age_in_bounds = all(
        (true_age < (np.percentile(preds.posterior_predictive['ages'],97.5,axis=1).flatten()))
        & (true_age > (np.percentile(preds.posterior_predictive['ages'],2.5,axis=1).flatten()))
    )

    assert all([d13c_in_bounds, age_in_bounds]), f"err: {d13c_in_bounds} d13C bounds check or {age_in_bounds} age bounds check"


# def lengthscale_stability()

# def proxy_inference_stability()
    