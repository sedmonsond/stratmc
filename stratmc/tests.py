import numpy as np

from stratmc.data import clean_data


def check_inference(full_trace, sample_df, ages_df, quiet = True, **kwargs):
    """
    Master function (calls each of the functions in the ``tests`` module) for checking that superposition is never violated in the posterior. Returns a list of chain indices where superposition was violated; these chains can be dropped from the trace using :py:meth:`drop_chains() <stratmc.data.drop_chains>`. Run automatically inside of :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    Because of the likelihood penalty used to manually enforce detrital and intrusive ages in :py:meth:`intermediate_detrital_potential() <stratmc.model.intermediate_detrital_potential>` and :py:meth:`intermediate_intrusive_potential() <stratmc.model.intermediate_intrusive_potential>` (called in :py:meth:`build_model() <stratmc.model.build_model>`), rare chains may have minor superposition violations when deterital/intrusive ages are present. These chains can simply be discarded. If superposition is frequently violated in a given section, or if superposition violations are severe, check that the heights for all age constraints in ``ages_df`` are correct, and that the reported ages respect superposition. The model can correct for mean ages that are out of superposition, but may fail if the age constraints do not overlap given their 2$\sigma$ uncertainties.

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    sections: list(str) or numpy.array(str), optional
        List of sections included in the inference. Defaults to all sections in ``sample_df``.

    quiet: bool, optional
        Whether to print the type, section name, and chain/draw of each superposition violation; defaults to ``False``.

    Returns
    -------
    bad_chains: numpy.array
        Array of chain indices where superposition was violated in the posterior.

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

    bad_chains_1 = check_superposition(full_trace, sample_df, ages_df, sections = sections, quiet = quiet)
    bad_chains_2 = check_detrital_ages(full_trace, sample_df, ages_df, sections = sections, quiet = quiet)
    bad_chains_3 = check_intrusive_ages(full_trace, sample_df, ages_df,sections = sections, quiet = quiet)

    bad_chains = np.concatenate([bad_chains_1, bad_chains_2, bad_chains_3])

    return np.unique(bad_chains)


def check_superposition(full_trace, sample_df, ages_df, quiet = True, **kwargs):
    """
    Check that stratigraphic superposition between all age constriants and samples is respected in the posterior.

    Parameters
    ----------
    full_trace: arviz.InferenceData
        An :class:`arviz.InferenceData` object containing the full set of prior and posterior samples from :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections.

    sections: list(str) or numpy.array(str), optional
        List of sections included in the inference. Defaults to all sections in ``sample_df``.

    quiet: bool, optional
        Whether to print the section name and chain/draw of each superposition violation; defaults to ``False``.

    Returns
    -------
    bad_chains: numpy.array
        Array of chain indices where superposition was violated in the posterior.

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

    ages_df = ages_df[(~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?'])]

    chains = list(full_trace.posterior.chain.values)
    bad_chains = []
    for section in sections:
        sample_heights = sample_df[sample_df['section']==section]['height']
        age_heights = ages_df[ages_df['section']==section]['height']

        comb = np.concatenate([sample_heights, age_heights])
        sort_idx = np.argsort(comb)

        # chains x draws x # samples
        for c, chain in enumerate(chains):
            sample_posterior = np.swapaxes(full_trace.posterior[str(section) + '_ages'].values[c, :, :], 0, 1)
            # chains x draws x # ages
            age_posterior = np.swapaxes(full_trace.posterior[str(section) + '_radiometric_age'].values[c, :, :], 0, 1)

            posterior_stacked = np.vstack([sample_posterior, age_posterior])

            draws = posterior_stacked.shape[1]

            for i in range(draws):
                #assert(all(np.diff(posterior_stacked[sort_idx,i].ravel()) <= 0)), "stratigraphic superposition violated in section " + str(section) + " draw " + str(i)
                if not all(np.diff(posterior_stacked[sort_idx,i].ravel()) <= 0):
                    if not quiet:
                        print("stratigraphic superposition violated in section " + str(section) + ', chain ' + str(chain) +  ", draw " + str(i) + '. Check that the heights for all age constraints in ``ages_df`` are correct, and that the reported ages respect superposition (the model can correct for mean ages that are out of superposition, but may fail if the age constraints do not overlap given their reported uncertainties).')
                    bad_chains.append(chain)

    bad_chains = np.unique(bad_chains)

    return bad_chains


def check_detrital_ages(full_trace, sample_df, ages_df, quiet = True, **kwargs):
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

    sections: list(str) or numpy.array(str), optional
        List of sections included in the inference. Defaults to all sections in ``sample_df``.

    quiet: bool, optional
        Whether to print the section name and chain/draw of each superposition violation; defaults to ``False``.

    Returns
    -------
    bad_chains: numpy.array
        Array of chain indices where superposition was violated in the posterior.

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


    chains = list(full_trace.posterior.chain.values)
    bad_chains = []

    for section in sections:
        section_df = sample_df[(sample_df['section']==section) & (~sample_df['Exclude?'])]
        section_df.sort_values(by = 'height', inplace = True)

        section_age_df = ages_df[(ages_df['section']==section) & (~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?']) & ~(ages_df['Exclude?'])]
        intermediate_detrital_section_ages_df = ages_df[(ages_df['section']==section) & (ages_df['intermediate detrital?'])]

        age_heights = section_age_df['height'].values

        post_section_ages = full_trace.posterior[section + '_ages'].values

        for c, chain in enumerate(chains):
            section_sample_ages = np.swapaxes(post_section_ages[c, :, :], 0, 1)

            for interval in np.arange(0, len(age_heights)-1).tolist():

                above = intermediate_detrital_section_ages_df['height']>age_heights[interval]
                below = intermediate_detrital_section_ages_df['height']<age_heights[interval+1]
                detrital_interval_df = intermediate_detrital_section_ages_df[above & below]

                for height, shared, name, i in zip (detrital_interval_df['height'], detrital_interval_df['shared?'], detrital_interval_df['name'], np.arange(detrital_interval_df['height'].shape[0])):
                    sample_idx = section_df['height'] > height

                    # check that all the posterior ages for overlying samples are younger than the detrital age during each draw
                    if shared:
                        detrital_age_posterior = full_trace.posterior[name].values[c, :]
                    else:
                        dist_name = str(section)+'_'+ str(interval) +'_' + 'detrital_age_' + str(i)
                        detrital_age_posterior = full_trace.posterior[dist_name].values[c, :]


                    for draw in np.arange(section_sample_ages.shape[1]):
                        #assert all(section_sample_ages[:, draw][sample_idx] <= detrital_age_posterior[draw])
                        if not all(section_sample_ages[:, draw][sample_idx] <= detrital_age_posterior[draw]):
                            if not quiet:
                                print('Detrital age constraint violated in section ' + str(section) + ', chain ' + str(chain) +  ", draw " + str(draw))
                            bad_chains.append(chain)

    bad_chains = np.unique(bad_chains)

    return bad_chains

def check_intrusive_ages(full_trace, sample_df, ages_df, quiet = True, **kwargs):
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

    sections: list(str) or numpy.array(str), optional
        List of sections included in the inference. Defaults to all sections in ``sample_df``.

    quiet: bool, optional
        Whether to print the section name and chain/draw of each superposition violation; defaults to ``False``.

    Returns
    -------
    bad_chains: numpy.array
        Array of chain indices where superposition was violated in the posterior.

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

    chains = list(full_trace.posterior.chain.values)
    bad_chains = []

    for section in sections:
        section_df = sample_df[sample_df['section']==section]
        section_df.sort_values(by = 'height', inplace = True)

        section_age_df = ages_df[(ages_df['section']==section) & (~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?']) & ~(ages_df['Exclude?'])]
        intermediate_intrusive_section_ages_df = ages_df[(ages_df['section']==section) & (ages_df['intermediate intrusive?'])]

        age_heights = section_age_df['height'].values

        post_section_ages = full_trace.posterior[section + '_ages'].values

        for c, chain in enumerate(chains):
            section_sample_ages = np.swapaxes(post_section_ages[c, :, :], 0, 1)

            for interval in np.arange(0, len(age_heights)-1).tolist():

                above = intermediate_intrusive_section_ages_df['height']>age_heights[interval]
                below = intermediate_intrusive_section_ages_df['height']<age_heights[interval+1]
                intrusive_interval_df = intermediate_intrusive_section_ages_df[above & below]

                for height, shared, name, i in zip (intrusive_interval_df['height'], intrusive_interval_df['shared?'], intrusive_interval_df['name'], np.arange(intrusive_interval_df['height'].shape[0])):
                    sample_idx = section_df['height'] < height

                    # check that all the posterior ages for overlying samples are younger than the detrital age during each draw
                    if shared:
                        intrusive_age_posterior = full_trace.posterior[name].values[c, :]
                    else:
                        dist_name = str(section)+'_'+ str(interval) +'_' + 'intrusive_age_' + str(i)
                        intrusive_age_posterior = full_trace.posterior[dist_name].values[c, :]

                    for draw in np.arange(section_sample_ages.shape[1]):
                        # assert all(section_sample_ages[:, draw][sample_idx] >= intrusive_age_posterior[draw])
                        if not all(section_sample_ages[:, draw][sample_idx] >= intrusive_age_posterior[draw]):
                            if not quiet:
                                print('Intrusive age constraint violated in section '  + str(section) + ', chain ' + str(chain) +  ", draw " + str(draw))
                            bad_chains.append(chain)

    bad_chains = np.unique(bad_chains)

    return bad_chains
