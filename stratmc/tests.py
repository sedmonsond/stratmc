import numpy as np

from stratmc.data import clean_data


def check_inference(full_trace, sample_df, ages_df, mode = 'posterior', quiet = True, **kwargs):
    """
    Master function (calls each of the functions in the ``tests`` module) for checking that superposition is never violated in the posterior (or prior). Returns a list of chain indices where superposition was violated; these chains can be dropped from the trace using :py:meth:`drop_chains() <stratmc.data.drop_chains>`. Run automatically inside of :py:meth:`get_trace() <stratmc.inference.get_trace>` in :py:mod:`stratmc.inference`.

    Because of the likelihood penalty used to manually enforce detrital and intrusive ages in :py:meth:`intermediate_detrital_potential() <stratmc.model.intermediate_detrital_potential>` and :py:meth:`intermediate_intrusive_potential() <stratmc.model.intermediate_intrusive_potential>` (called in :py:meth:`build_model() <stratmc.model.build_model>`), rare chains may have minor superposition violations when deterital/intrusive ages are present. These chains can simply be discarded. If superposition is frequently violated in a given section, or if superposition violations are severe, check that the heights for all age constraints in ``ages_df`` are correct, and that the reported ages respect superposition. The model can correct for mean ages that are out of superposition, but may fail if the age constraints do not overlap within uncertainty.

    NOTE: this can work incorrectly if samples marked 'Exclude?' and 'superposition? = True' were left in the DataFrame, and multiple samples come from the same stratigraphic height.

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

    #get list of proxies included in model from full_trace
    if 'proxies' in kwargs:
        proxies = kwargs['proxies']
    else:
        variables = [
                l
                for l in list(full_trace[mode].data_vars.keys())
                if (f"{'gp_ls_'}" in l) and (f"{'unshifted'}" not in l)
                ]

        proxies = []
        for var in variables:
            proxies.append(var[6:])

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = np.unique(sample_df.dropna(subset = proxies, how = 'all')['section'])

    bad_chains_1, bad_draws_1, bad_draws_per_section_1 = check_superposition(full_trace, sample_df, ages_df, sections = sections, mode = mode, proxies = proxies, quiet = quiet)
    bad_chains_2, bad_draws_2, bad_draws_per_section_2 = check_detrital_ages(full_trace, sample_df, ages_df, sections = sections, mode = mode, proxies = proxies, quiet = quiet)
    bad_chains_3, bad_draws_3, bad_draws_per_section_3 = check_intrusive_ages(full_trace, sample_df, ages_df,sections = sections, mode = mode, proxies = proxies, quiet = quiet)

    bad_chains = np.concatenate([bad_chains_1, bad_chains_2, bad_chains_3])

    chains = list(full_trace[mode].chain.values)

    bad_draws = {}
    for chain in chains:
        bad_draws[chain] = np.concatenate([bad_draws_1[chain], bad_draws_2[chain], bad_draws_3[chain]])
        bad_draws[chain] = np.unique(bad_draws[chain])

    bad_draws_per_section = {}
    for section in sections:
        bad_draws_per_section[section] = {}
        for chain in chains:
            bad_draws_per_section[section][chain] = np.concatenate([bad_draws_per_section_1[section][chain], bad_draws_per_section_2[section][chain], bad_draws_per_section_3[section][chain]])

            bad_draws_per_section[section][chain]  = np.unique(bad_draws_per_section[section][chain])


    return np.unique(bad_chains), bad_draws, bad_draws_per_section

def check_superposition(full_trace, sample_df, ages_df, mode = 'posterior', quiet = True, **kwargs):
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
    if 'proxies' in kwargs:
        proxies = kwargs['proxies']
    else:
        variables = [
                l
                for l in list(full_trace[mode].data_vars.keys())
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

    # check that for all constraints with 'shared == True', the constraint actually is used >1 time (if not, set shared = False)
    for shared_age_name in ages_df[ages_df['shared?']==True]['name']:
        if ages_df[(ages_df['shared?'] == True) & (ages_df['name']==shared_age_name)].shape[0] < 2:
            idx = ages_df[(ages_df['shared?'] == True) & (ages_df['name']==shared_age_name)].index
            ages_df.loc[idx, 'shared?'] = False

    ages_df = ages_df[(~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?']) & (~ages_df['depositional?'])]

    chains = list(full_trace[mode].chain.values)
    bad_chains = []
    bad_draws = {}
    bad_draws_per_section = {}

    for chain in chains:
        bad_draws[chain] = []

    for section in sections:
        bad_draws_per_section[section] = {}
        section_df = sample_df[sample_df['section'] == section]

        sample_superposition = section_df['superposition?'].values

        shuffle_heights = np.unique(section_df['height'].values[~sample_superposition])

        sample_heights = section_df['height'][sample_superposition]
        age_heights = ages_df[ages_df['section']==section]['height']

        comb = np.concatenate([sample_heights, age_heights])
        sort_idx = np.argsort(comb)

        # check superposition for samples w/ superposition = True
        # chains x draws x # samples
        for c, chain in enumerate(chains):
            bad_draws_per_section[section][chain] = []
            sample_posterior = np.swapaxes(full_trace[mode][str(section) + '_ages'].values[c, :, :], 0, 1)[sample_superposition, :]
            # chains x draws x # ages
            age_posterior = np.swapaxes(full_trace[mode][str(section) + '_radiometric_age'].values[c, :, :], 0, 1)

            posterior_stacked = np.vstack([sample_posterior, age_posterior])

            draws = posterior_stacked.shape[1]
            for i in range(draws):
                #assert(all(np.diff(posterior_stacked[sort_idx,i].ravel()) <= 0)), "stratigraphic superposition violated in section " + str(section) + " draw " + str(i)
                if not all(np.diff(posterior_stacked[sort_idx,i].ravel()) <= 0):
                    if not quiet:
                        print("stratigraphic superposition violated in section " + str(section) + ', chain ' + str(chain) +  ", draw " + str(i) + '. Check that the heights for all age constraints in ``ages_df`` are correct, and that the reported ages respect superposition (the model can correct for mean ages that are out of superposition, but may fail if the age constraints do not overlap given their reported uncertainties).')
                    bad_chains.append(chain)
                    bad_draws[chain].append(i)
                    bad_draws_per_section[section][chain].append(i)

            # check superposition for samples w/ superposition = False
            for h in shuffle_heights:
                older_idx = section_df['height'] < h
                younger_idx = section_df['height'] > h

                older_ages = np.swapaxes(full_trace[mode][str(section) + '_ages'].values[c, :, :], 0, 1)[older_idx, :]
                younger_ages = np.swapaxes(full_trace[mode][str(section) + '_ages'].values[c, :, :], 0, 1)[younger_idx, :]

                shuffle_sample_idx = np.where((~section_df['superposition?']) & (~section_df['Exclude?']) & (section_df['height'] == h))[0]

                # check each sample in the shuffled group
                for idx in shuffle_sample_idx:
                    current_age = np.swapaxes(full_trace[mode][str(section) + '_ages'].values[c, :, :], 0, 1)[idx, :]

                    # check each draw
                    for i in range(draws):
                        if not (all(older_ages[:, i] >= current_age[i])) and (all(younger_ages[:, i] <= current_age[i])):
                            if not quiet:
                                print("stratigraphic superposition violated in section " + str(section) + ', chain ' + str(chain) +  ", draw " + str(i) + '. Check that the heights for all age constraints in ``ages_df`` are correct, and that the reported ages respect superposition (the model can correct for mean ages that are out of superposition, but may fail if the age constraints do not overlap given their reported uncertainties).')

                            bad_chains.append(chain)
                            bad_draws[chain].append(i)
                            bad_draws_per_section[section][chain].append(i)

            bad_draws_per_section[section][chain] = np.unique(bad_draws_per_section[section][chain])

    for chain in chains:
        bad_draws[chain] = np.unique(bad_draws[chain])

    bad_chains = np.unique(bad_chains)

    return bad_chains, bad_draws, bad_draws_per_section

def check_detrital_ages(full_trace, sample_df, ages_df, quiet = True, mode = 'posterior', **kwargs):
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
    if 'proxies' in kwargs:
        proxies = kwargs['proxies']
    else:
        variables = [
                l
                for l in list(full_trace[mode].data_vars.keys())
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

    # check that for all constraints with 'shared == True', the constraint actually is used >1 time (if not, set shared = False)
    for shared_age_name in ages_df[ages_df['shared?']==True]['name']:
        if ages_df[(ages_df['shared?'] == True) & (ages_df['name']==shared_age_name)].shape[0] < 2:
            idx = ages_df[(ages_df['shared?'] == True) & (ages_df['name']==shared_age_name)].index
            ages_df.loc[idx, 'shared?'] = False


    chains = list(full_trace[mode].chain.values)
    bad_chains = []
    bad_draws = {}
    bad_draws_per_section = {}

    for chain in chains:
        bad_draws[chain] = []

    for section in sections:
        bad_draws_per_section[section] = {}
        section_df = sample_df[(sample_df['section']==section)]
        section_df.sort_values(by = 'height', inplace = True)

        section_age_df = ages_df[(ages_df['section']==section) & (~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?'])  & (~ages_df['depositional?']) & ~(ages_df['Exclude?'])]
        intermediate_detrital_section_ages_df = ages_df[(ages_df['section']==section) & (ages_df['intermediate detrital?'])]

        age_heights = section_age_df['height'].values

        post_section_ages = full_trace[mode][section + '_ages'].values

        # shape = chains x draws x ages
        post_section_radio_ages = full_trace[mode][section + '_radiometric_age'].values

        for c, chain in enumerate(chains):
            bad_draws_per_section[section][chain] = []
            # shape: samples x draws
            section_sample_ages = np.swapaxes(post_section_ages[c, :, :], 0, 1)

            # shape: age constraints x draws
            section_radio_ages = np.swapaxes(post_section_radio_ages[c, :, :], 0, 1)

            for interval in np.arange(0, len(age_heights)-1).tolist():

                above = intermediate_detrital_section_ages_df['height']>age_heights[interval]
                below = intermediate_detrital_section_ages_df['height']<age_heights[interval+1]
                detrital_interval_df = intermediate_detrital_section_ages_df[above & below]


                above = section_df['height']>=age_heights[interval]
                below = section_df['height']<age_heights[interval+1]
                interval_df = section_df[above & below]

                for height, shared, name, i in zip (detrital_interval_df['height'], detrital_interval_df['shared?'], detrital_interval_df['name'], np.arange(detrital_interval_df['height'].shape[0])):
                    sample_idx = section_df['height'] > height

                    if len(interval_df[interval_df['height']>=detrital_interval_df['height'].values[i]]['height'].values)>0:
                        # check that all the posterior ages for overlying samples are younger than the detrital age during each draw
                        if shared:
                            detrital_age_posterior = full_trace[mode][name].values[c, :]
                        else:
                            dist_name = str(section)+'_'+ str(interval) +'_' + 'detrital_age_' + str(i)
                            detrital_age_posterior = full_trace[mode][dist_name].values[c, :]


                        for draw in np.arange(section_sample_ages.shape[1]):
                            #assert all(section_sample_ages[:, draw][sample_idx] <= detrital_age_posterior[draw])
                            if not all(section_sample_ages[:, draw][sample_idx] <= detrital_age_posterior[draw]):
                                if not quiet:
                                    print('Detrital age constraint violated in section ' + str(section) + ', chain ' + str(chain) +  ", draw " + str(draw))
                                bad_chains.append(chain)
                                bad_draws[chain].append(draw)
                                bad_draws_per_section[section][chain].append(draw)

                            # check superposition between each detrital age and overlying depositional age
                            if not(section_radio_ages[interval + 1, draw] <= detrital_age_posterior[draw]):
                                if not quiet:
                                    print('Superposition between depositional and detrital age constraint violated in ' + str(section) + ', chain ' + str(chain) +  ", draw " + str(draw))
                                bad_chains.append(chain)
                                bad_draws[chain].append(draw)
                                bad_draws_per_section[section][chain].append(draw)


            bad_draws_per_section[section][chain] = np.unique(bad_draws_per_section[section][chain])


    for chain in chains:
        bad_draws[chain] = np.unique(bad_draws[chain])

    bad_chains = np.unique(bad_chains)

    return bad_chains, bad_draws, bad_draws_per_section

def check_intrusive_ages(full_trace, sample_df, ages_df, mode = 'posterior', quiet = True, **kwargs):
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
    if 'proxies' in kwargs:
        proxies = kwargs['proxies']
    else:
        variables = [
                l
                for l in list(full_trace[mode].data_vars.keys())
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

    # check that for all constraints with 'shared == True', the constraint actually is used >1 time (if not, set shared = False)
    for shared_age_name in ages_df[ages_df['shared?']==True]['name']:
        if ages_df[(ages_df['shared?'] == True) & (ages_df['name']==shared_age_name)].shape[0] < 2:
            idx = ages_df[(ages_df['shared?'] == True) & (ages_df['name']==shared_age_name)].index
            ages_df.loc[idx, 'shared?'] = False

    chains = list(full_trace[mode].chain.values)
    bad_chains = []
    bad_draws = {}
    bad_draws_per_section = {}

    for chain in chains:
        bad_draws[chain] = []

    for section in sections:
        bad_draws_per_section[section] = {}

        section_df = sample_df[sample_df['section']==section]
        section_df.sort_values(by = 'height', inplace = True)

        section_age_df = ages_df[(ages_df['section']==section) & (~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?'])  & (~ages_df['depositional?']) & ~(ages_df['Exclude?'])]
        intermediate_intrusive_section_ages_df = ages_df[(ages_df['section']==section) & (ages_df['intermediate intrusive?'])]

        age_heights = section_age_df['height'].values

        post_section_ages = full_trace[mode][section + '_ages'].values

        # shape = chains x draws x ages
        post_section_radio_ages = full_trace[mode][section + '_radiometric_age'].values

        for c, chain in enumerate(chains):
            bad_draws_per_section[section][chain] = []

            # shape: samples x draws
            section_sample_ages = np.swapaxes(post_section_ages[c, :, :], 0, 1)

            # shape: age constraints x draws
            section_radio_ages = np.swapaxes(post_section_radio_ages[c, :, :], 0, 1)

            for interval in np.arange(0, len(age_heights)-1).tolist():

                above = intermediate_intrusive_section_ages_df['height']>age_heights[interval]
                below = intermediate_intrusive_section_ages_df['height']<age_heights[interval+1]
                intrusive_interval_df = intermediate_intrusive_section_ages_df[above & below]

                for height, shared, name, i in zip (intrusive_interval_df['height'], intrusive_interval_df['shared?'], intrusive_interval_df['name'], np.arange(intrusive_interval_df['height'].shape[0])):
                    sample_idx = section_df['height'] < height

                    # check that all the posterior ages for overlying samples are younger than the detrital age during each draw
                    if shared:
                        intrusive_age_posterior = full_trace[mode][name].values[c, :]
                    else:
                        dist_name = str(section)+'_'+ str(interval) +'_' + 'intrusive_age_' + str(i)
                        intrusive_age_posterior = full_trace[mode][dist_name].values[c, :]

                    for draw in np.arange(section_sample_ages.shape[1]):
                        # assert all(section_sample_ages[:, draw][sample_idx] >= intrusive_age_posterior[draw])
                        if not all(section_sample_ages[:, draw][sample_idx] >= intrusive_age_posterior[draw]):
                            if not quiet:
                                print('Intrusive age constraint violated in section '  + str(section) + ', chain ' + str(chain) +  ", draw " + str(draw))
                            bad_chains.append(chain)
                            bad_draws[chain].append(draw)
                            bad_draws_per_section[section][chain].append(draw)

                        # check superposition between each intrusive age and underlying depositional age
                        if not(section_radio_ages[interval, draw] >= intrusive_age_posterior[draw]):
                            if not quiet:
                                print('Superposition between depositional and intrusive age constraint violated in ' + str(section) + ', chain ' + str(chain) +  ", draw " + str(draw))
                            bad_chains.append(chain)
                            bad_draws[chain].append(draw)
                            bad_draws_per_section[section][chain].append(draw)

            bad_draws_per_section[section][chain] = np.unique(bad_draws_per_section[section][chain])

    for chain in chains:
        bad_draws[chain] = np.unique(bad_draws[chain])

    bad_chains = np.unique(bad_chains)

    return bad_chains, bad_draws, bad_draws_per_section
