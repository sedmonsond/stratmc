import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pymc as pm
import pymc.distributions.transforms as tr
import pytensor.tensor as at
from pytensor import shared
from pytensor.link.jax.dispatch import jax_funcify

# wrapper for pytensor tensor sorting
from pytensor.tensor.sort import SortOp

from stratmc.data import clean_data


@jax_funcify.register(SortOp)
def jax_funcify_SortOp(op, node, **kwargs):

    def sort(a, axis=-1, stable=True, order=None):
        return jnp.sort(a, axis = -1, stable = True)

    return sort

DIST_DICT = {
    "Wald": pm.Wald,
    "Normal": pm.Normal,
    "HalfFlat": pm.HalfFlat,
    "HalfCauchy": pm.HalfCauchy,
    "Uniform": pm.Uniform,
    "Flat": pm.Flat,
    "Beta": pm.Beta,
    "Exponential": pm.Exponential,
    "StudentT": pm.StudentT,
    "Cauchy": pm.Cauchy,
    "Laplace": pm.Laplace,
    "Kumaraswamy": pm.Kumaraswamy,
    "Weibull": pm.Weibull,
    "HalfStudentT": pm.HalfStudentT,
    "LogNormal": pm.LogNormal,
    "ChiSquared": pm.ChiSquared,
    "HalfNormal": pm.HalfNormal,
    "Pareto": pm.Pareto,
    "InverseGamma": pm.InverseGamma,
    "ExGaussian": pm.ExGaussian,
    "VonMises": pm.VonMises,
    "SkewNormal": pm.SkewNormal,
    "Triangular": pm.Triangular,
    "Gumbel": pm.Gumbel,
    "Logistic": pm.Logistic,
    "LogitNormal": pm.LogitNormal,
    "Rice": pm.Rice,
    "Moyal": pm.Moyal,
    "AsymmetricLaplace": pm.AsymmetricLaplace,
    "PolyaGamma": pm.PolyaGamma,
}

def build_model(sample_df, ages_df, proxies = ['d13c'], proxy_sigma_default = 0.1, approximate = False,  hsgp_m = 15, hsgp_c = 1.3, ls_dist = 'Wald', ls_min = 0, ls_mu = 20, ls_lambda = 50, ls_sigma = 50, var_sigma = 10,  white_noise_sigma = 1e-1, gp_mean_mu = None, gp_mean_sigma = None, offset_type = 'section', offset_prior = 'Laplace', offset_alpha = 0, offset_beta = 1, offset_sigma = 1, offset_mu = 0, offset_b = 2, noise_type = 'section', noise_prior = 'HalfCauchy', noise_beta = 1, noise_sigma = 1, noise_nu = 1,  jitter = 0.001, proxy_observed = True, **kwargs):
    """
    Create a proxy signal (i.e., carbon isotope) :ref:`inference model <model_target>`.

    Note that while excluded samples (``Exclude?`` is ``True`` in ``sample_df``) do not affect the proxy signal reconstruction, their ages are passively tracked within the inference model (if other samples from the same sections are included).

    Parameters
    ----------
    sample_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing proxy data for all sections. Load from .csv file using :py:meth:`load_data() <stratmc.data.load_data>` in :py:mod:`stratmc.data`.

    ages_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for all sections. Load from .csv file using :py:meth:`load_data() <stratmc.data.load_data>` in :py:mod:`stratmc.data`.

    sections:: list(str) or numpy.array(str), optional
        List of sections to include in the inference model. Defaults to all sections in ``sample_df``.

    proxies: str or list(str), optional
        Column or columns containing proxy data in ``sample_df``. Defaults to 'd13c'.

    proxy_sigma_default: float or dict{float}, optional
        Measurement uncertainty (:math:`1\\sigma`) to use for proxy observations if not specified in ``proxy_std`` column of ``sample_df``. To set a different value for each proxy, pass a dictionary with proxy names as keys. Defaults to 0.1.

    approximate: bool, optional
        Build model with an unapproximated latent GP (:class:`pymc.gp.Latent`) if ``False``, or a Hilbert space Gaussian process approximation (:class:`pymc.gp.HSGP`) if ``True``; defaults to ``False``. If using the HSGP approximation, also pass the ``hsgp_m`` and ``hsgp_c`` parameters (the defaults are unlikely to work well for all problems). Appropriate values for ``m`` and ``c`` can be estimated using :py:meth:`approx_hsgp_hyperparams() <pymc.gp.hsgp_approx.approx_hsgp_hyperparams>`.

    hsgp_m: int or dict{int}, optional
        Number of basis vectors to use in the HSGP approximation (see :class:`pymc.gp.HSGP`). Pass as a dictionary with proxy names as keys to specify a different value for each proxy. Defaults to 15.

    hsgp_c: float or dict{float}, optional
        Proportion extension factor for the HSGP approximation (see :class:`pymc.gp.HSGP`). Pass as a dictionary with proxy names as keys to specify a different value for each proxy. Defaults to 1.3.

    ls_dist: str or dict{str}, optional
        Prior distribution for the lengthscale hyperparameter of the exponential quadratic covariance kernel (:class:`pymc.gp.cov.ExpQuad <pymc.gp.cov.ExpQuad>`); set to ``Wald`` (:class:`pymc.Wald`) or ``HalfNormal`` (:class:`pymc.HalfNormal`). Defaults to ``Wald`` with ``mu = 20`` and ``lambda = 50``. To change ``mu`` and ``lambda``, pass the ``ls_mu`` and ``ls_lambda`` parameters. For ``HalfNormal``, the variance defaults to ``sigma = 50``; change by passing ``ls_sigma``. Pass as a dictionary with proxy names as keys to specify a different prior distribution for each proxy.

    ls_min: float or dict{float}, optional
        Minimum value for the lengthscale hyperparameter of the :class:`pymc.gp.cov.ExpQuad` covariance kernel; shifts the lengthscale prior by ``ls_min``. Pass as a dictionary with proxy names as keys to specify a different value for each proxy. Defaults to 0.

    ls_mu: float or dict{float}, optional
        Mean (`mu`) of the :class:`pymc.gp.cov.ExpQuad` lengthscale prior if ``ls_dist = `Wald```. Pass as a dictionary with proxy names as keys to specify a different value for each proxy. Defaults to 20.

    ls_lambda: float or dict{float}, optional
        Relative precision (`lam`) of the :class:`pymc.gp.cov.ExpQuad` lengthscale hyperparameter prior if ``ls_dist = `Wald```. Pass as a dictionary with proxy names as keys to specify a different value for each proxy. Defaults to 50.

    ls_sigma: float or dict{float}, optional
        Scale parameter (`sigma`) of the :class:`pymc.gp.cov.ExpQuad` lengthscale hyperparameter prior if ``ls_dist = `HalfNormal```. Pass as a dictionary with proxy names as keys to specify a different value for each proxy. Defaults to 50.

    var_sigma: float or dict{float}, optional
        Scale parameter (`sigma`) of the covariance kernel variance hyperparameter prior, which is a :class:`pymc.HalfNormal` distribution. Pass as a dictionary with proxy names as keys to specify a different value for each proxy. Defaults to 10.

    white_noise_sigma: float or dict{float}, optional
        Amplitude of white noise component of GP covariance function. Defaults to 0.1. Should be equal to or less than the proxy measurement uncertainty; use a smaller value for proxies with low-amplitude signals (e.g., Sr isotopes). Pass as a dictionary with proxy names as keys to specify a different value for each proxy.

    gp_mean_mu: float or dict{float}, optional
        Mean (`mu`) of the GP mean function prior, which is a :class:`pymc.Normal` distribution. Defaults to the mean of the observations for each proxy. Pass as a dictionary with proxy names as keys to specify a different value for each proxy.

    gp_mean_sigma: float or dict{float}, optional
        Standard deviation (`sigma`) of the GP mean function prior, which is a :class:`pymc.Normal` distribution. Defaults to twice the standard deviation of the observations for each proxy. Pass as a dictionary with proxy names as keys to specify a different value for each proxy.

    offset_type: str or dict{str}, optional
        Parameterize offset such that all samples from the same section have the same offset (set to ``section``), or such that custom sample groups share an offset term (set to ``groups``, and specify the group for each sample and proxy with a ``offset_group_proxy`` column in ``sample_df``). To omit the offset term, set to ``none``. Defaults to ``section``. Pass as a dictionary with proxy names as keys to specify a different offset type for each proxy.

    offset_prior: str or dict{str}, optional
        Type of distribution to use for the offset prior. Pass as a ``string`` to use the same prior for all proxies, or as a ``dict`` of ``string`` (with proxy names as keys) to specify a different prior distribution for each proxy. Defaults to ``Laplace`` (:class:`pymc.Laplace`) with ``mu = 0`` and ``b = 2``. ``mu`` and ``b`` can be changed by passing ``offset_mu`` and ``offset_b``. To use other types of priors, pass the name of a distribution from :class:`pymc.distributions`, along with parameter values in ``offset_params``. Pass as a dictionary with proxy names as keys to specify a different prior for each proxy.

    offset_params: dict{float} or dict{dict{float}}, optional
        Only required if using a custom ``offset_prior``. Pass as a dictionary to use the same parameters for all proxies, or as a dictionary of dictionaries (with proxy names as keys) to specify different parameters for each proxy. Keys are ``param_1_name``, ``param_1``, ``param_2_name``, and ``param_2`` (with parameter names corresponding to those required for the specified :class:`pymc.distributions` object). If only one parameter is required, use ``np.nan`` for ``param_2_name`` and ``param_2``.

    noise_type: str or dict{str}, optional
        Parameterize noise as per-section (set to ``section``) or per-group (set to ``groups``, and specify the group for each sample and proxy with a ``noise_group_proxy`` column in ``sample_df``). Defaults to ``section``. Pass as a dictionary with proxy names as keys to specify a different noise type for each proxy.

    noise_prior: str or dict{str}, optional
        Type of distribution to use for the noise prior. Pass as a ``string`` to use the same prior for all proxies, or as a ``dict`` of ``string`` (with proxy names as keys) to specify a different prior distribution for each proxy. Defaults to ``HalfCauchy`` (:class:`pymc.HalfCauchy`) with ``beta = 1``. ``beta`` can be changed by passing ``noise_beta``. Other implemented priors (note that noise must be positive-only) are ``HalfNormal`` (:class:`pymc.HalfNormal`; defaults to ``noise_sigma = 1``) and ``HalfStudentT`` (:class:`pymc.HalfStudentT`; defaults to ``noise_nu = 1`` and ``noise_sigma`` = 1).

    superposition_dict: dict{list(str)}, optional
        Optional; dictionary specifying superposition relationships between different sections. Should only be used when superposition is not implicitly enforced by the age constraints; for example, when sections share the same minimum and maximum age constraints, but are from geological formations with a known stratigraphic relationship. Dictionary keys are section names, and the value for each key is a list of sections that must be older (stratigraphically lower).

    jitter: float, optional
        Value of ``jitter`` passed to :meth:`pymc.gp.Latent.prior`. Defaults to 0.001.

    proxy_observed: bool, optional
        Whether to pass observed values to the likelihood function; defaults to ``True``. Only set to ``False`` to generate synthetic observations from the model prior in :py:meth:`synthetic_observations_from_prior() <stratmc.synthetics>` in :py:mod:`stratmc.synthetics`.

    Returns
    -------
    model: PyMC model
        :class:`pymc.model.core.Model` object.

    gp: pymc.gp.Latent or pymc.gp.HSGP
        Gaussian process prior for the model. :class:`pymc.gp.Latent` if ``approximate = True``, or :class:`pymc.gp.HSGP` if ``approximate = False``.
    """


    if type(proxies) == str:
        proxies = list([proxies])

    if type(proxy_sigma_default) != dict: #(type(proxy_sigma_default) == float) or (type(proxy_sigma_default) == int):
        temp = proxy_sigma_default
        proxy_sigma_default = {}
        for proxy in proxies:
            proxy_sigma_default[proxy] = temp

    # if offset_prior passed as a string, reformat as a dictionary (keys = proxies)
    if type(offset_prior) != dict:
        temp = offset_prior
        offset_prior = {}
        for proxy in proxies:
            offset_prior[proxy] = temp

    # if offset_type passed as a string, reformat as a dictionary (keys = proxies)
    if type(offset_type) != dict:
        temp = offset_type
        offset_type = {}
        for proxy in proxies:
            offset_type[proxy] = temp

    for proxy in proxies:
        if offset_type[proxy] not in ['section', 'groups', 'none']:
            sys.exit(f"offset type {offset_type[proxy]} not implemented. Choose from 'section' or 'groups'.")

    # also convert parameters for prior distributions to dictionaries
    if type(offset_alpha) != dict:
        temp = offset_alpha
        offset_alpha = {}
        for proxy in proxies:
            offset_alpha[proxy] = temp

    if type(offset_beta) != dict:
        temp = offset_beta
        offset_beta = {}
        for proxy in proxies:
            offset_beta[proxy] = temp

    if type(offset_sigma) != dict:
        temp = offset_sigma
        offset_sigma = {}
        for proxy in proxies:
            offset_sigma[proxy] = temp

    if type(offset_mu) != dict:
        temp = offset_mu
        offset_mu = {}
        for proxy in proxies:
            offset_mu[proxy] = temp

    if type(offset_b) != dict:
        temp = offset_b
        offset_b = {}
        for proxy in proxies:
            offset_b[proxy] = temp


    # if noise_prior passed as a string, reformat as a dictionary (keys = proxies)
    if type(noise_prior) != dict:
        temp = noise_prior
        noise_prior = {}
        for proxy in proxies:
            noise_prior[proxy] = temp

    # if noise_type passed as a string, reformat as a dictionary (keys = proxies)
    if type(noise_type) != dict:
        temp = noise_type
        noise_type = {}
        for proxy in proxies:
            noise_type[proxy] = temp

    for proxy in proxies:
        if noise_type[proxy] not in ['section', 'groups']:
            sys.exit(f"noise type {noise_type[proxy]} not implemented. Choose from 'section' or 'groups'.")

    # also convert noise prior parameters to dictionaries
    if type(noise_beta) != dict:
        temp = noise_beta
        noise_beta = {}
        for proxy in proxies:
            noise_beta[proxy] = temp

    if type(noise_sigma) != dict:
        temp = noise_sigma
        noise_sigma = {}
        for proxy in proxies:
            noise_sigma[proxy] = temp

    if type(noise_nu) != dict:
        temp = noise_nu
        noise_nu = {}
        for proxy in proxies:
            noise_nu[proxy] = temp

    # create dictionaries to store offset and noise terms inside of model
    offset_types = list(offset_type.values())
    if ('section' in offset_types) or  ('groups' in offset_types):
        offset_all = {}

    if 'groups' in offset_types:
        offset_group_dict = {}

    noise_types = list(noise_type.values())
    if ('section' in noise_types) or ('groups' in noise_types):
            noise_all = {}

    if 'groups' in noise_types:
        noise_group_dict = {}

    # if only one proxy, reformat kernel parameters as dictionaries
    if type(ls_dist) != dict:
        temp = ls_dist
        ls_dist = {}
        for proxy in proxies:
            ls_dist[proxy] = temp

    if type(ls_mu) != dict:
        temp = ls_mu
        ls_mu = {}
        for proxy in proxies:
            ls_mu[proxy] = temp

    if type(ls_lambda) != dict:
        temp = ls_lambda
        ls_lambda = {}
        for proxy in proxies:
            ls_lambda[proxy] = temp

    if type(ls_sigma) != dict:
        temp = ls_sigma
        ls_sigma = {}
        for proxy in proxies:
            ls_sigma[proxy] = temp

    if type(ls_min) != dict:
        temp = ls_min
        ls_min = {}
        for proxy in proxies:
            ls_min[proxy] = temp

    if type(var_sigma) != dict:
        temp = var_sigma
        var_sigma = {}
        for proxy in proxies:
            var_sigma[proxy] = temp

    if type(white_noise_sigma) != dict:
        temp = white_noise_sigma
        white_noise_sigma = {}
        for proxy in proxies:
            white_noise_sigma[proxy] = temp

    if gp_mean_mu is not None:
        if type(gp_mean_mu) != dict:
            temp = gp_mean_mu
            gp_mean_mu = {}
            for proxy in proxies:
                gp_mean_mu[proxy] = temp

    else:
        gp_mean_mu = {}
        for proxy in proxies:
            gp_mean_mu[proxy] = np.nanmean(sample_df[proxy].values)

    if gp_mean_sigma is not None:
         if type(gp_mean_sigma) != dict:
            temp = gp_mean_sigma
            gp_mean_sigma = {}
            for proxy in proxies:
                gp_mean_sigma[proxy] = temp

    else:
        gp_mean_sigma = {}
        for proxy in proxies:
            gp_mean_sigma[proxy] = 2 * np.nanstd(sample_df[proxy].values)

    # if using HSGP approximation, reformat m and c parameters
    if approximate:
        if type(hsgp_m) != dict:
            temp = hsgp_m
            hsgp_m = {}
            for proxy in proxies:
                hsgp_m[proxy] = temp

        if type(hsgp_c) != dict:
            temp = hsgp_c
            hsgp_c = {}
            for proxy in proxies:
                hsgp_c[proxy] = temp

    for proxy in proxies:
        if proxy + '_std' not in list(sample_df.columns):
            sample_df[proxy + '_std'] = np.nan

        idx = np.isnan(sample_df[proxy + '_std'])
        sample_df.loc[idx, proxy + '_std'] = proxy_sigma_default[proxy]

        if proxy + '_population_std' not in list(sample_df.columns):
            sample_df[proxy + '_population_std'] = np.nan

    if 'sections' in kwargs:
        sections = list(kwargs['sections'])
    else:
        sections = list(np.unique(sample_df['section']))

    if 'superposition_dict' in kwargs:
        superposition_dict = kwargs['superposition_dict']
    else:
        superposition_dict = {}

    for section in list(superposition_dict.keys()):
        for older_section in superposition_dict[section]:
            sections.remove(older_section)
            sections.insert(0, older_section)

    if 'offset_params' in kwargs:
        offset_params = kwargs['offset_params']

        # if not a dictionary, throw an error
        if type(offset_params) != dict:
            sys.exit(f"offset_params must be a dictionary")

        # if it's a dictionary, check that it has all the required items
        else:
            offset_keys = list(offset_params.keys())
            # if passed correctly, just reformat to assign same dict to each proxy
            if ('param_1_name' in offset_keys) & ('param_1' in offset_keys) &  ('param_2_name' in offset_keys) &  ('param_2' in offset_keys):
                temp_dict = {}
                for proxy in proxies:
                    temp_dict[proxy] = offset_params.copy()
                offset_params = temp_dict.copy()

            # if not passed correctly, and also not passed with proxies as keys
            elif all([proxy not in offset_keys for proxy in proxies]):

                sys.exit(f"offset_params must be a dictionary (or dictionary of dictionaries, if specifying different prior parameters for each proxy) with keys 'param_1_name','param_1', 'param_2_name', and 'param_2'")

            # if passed with proxies as keys
            else:
                for proxy in proxies:
                    # throw an error if every proxy doesn't have its own dictionary (unless using the default Laplace prior)
                    if proxy not in offset_keys:
                        # if using offset_mu and offset_b for the laplace prior, set prior_params dictionary to None
                        if offset_prior[proxy] == 'Laplace':
                            offset_params[proxy] = None
                        else:
                            sys.exit(f"offset_params must be a dictionary (or dictionary of dictionaries, if specifying different prior parameters for each proxy) with keys 'param_1_name','param_1', 'param_2_name', and 'param_2'. If passing with proxy names as keys, make sure that all parameters have been assigned to every proxy (except those using a Laplace prior with parameters offset_mu and offset_b).")


                    elif (proxy in offset_keys) and (offset_prior[proxy] == 'Laplace'):
                        print('Note: offset parameters passed in offset_params supersede the offset_mu and offset_b arguments')

                    # if the proxy has a dictionary, check that all the required parameters have been passed
                    else:
                        offset_proxy_keys =  list(offset_params[proxy].keys())
                        if ('param_1_name' not in offset_proxy_keys) | ('param_1' not in offset_proxy_keys) |  ('param_2_name' not in offset_proxy_keys) | ('param_2' not in offset_proxy_keys):
                             sys.exit(f"offset_params must be a dictionary (or dictionary of dictionaries, if specifying different prior parameters for each proxy) with keys 'param_1_name','param_1', 'param_2_name', and 'param_2'. If passing with proxy names as keys, make sure that all parameters have been assigned to every proxy.")
    else:
        offset_params = {}
        for proxy in proxies:
            offset_params[proxy] = None

    ## instead of removing samples from dataframe if they don't have any proxy observations, mark as excluded (so the model will still keep track of age at that height)
    # note - samples whose age shouldn't be tracked should simply be removed from the dataframe prior to running the inversion
    # for sample_df, 'exclude' now means exclude from the likelihood calculation, but keep track of age at that height
    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    # ignore sections that have no observations included in the inference (samples w/ no observations were marked `Exclude? = True` in clean_data)
    data_sections = list(np.unique(sample_df[~sample_df['Exclude?']]['section']))

    for section in list(sections):
        if section not in data_sections:
            sections.remove(section)

    # clean again (avoids issues w/ offset and noise groups)
    sample_df, ages_df = clean_data(sample_df, ages_df, proxies, sections)

    # check that for all constraints with 'shared == True', the constraint actually is used >1 time (if not, set shared = False)
    for shared_age_name in ages_df[ages_df['shared?']==True]['name']:
        if ages_df[(ages_df['shared?'] == True) & (ages_df['name']==shared_age_name)].shape[0] < 2:
            idx = ages_df[(ages_df['shared?'] == True) & (ages_df['name']==shared_age_name)].index
            ages_df.loc[idx, 'shared?'] = False

    with pm.Model() as model:

        gp_ls = {}
        gp_var = {}

        m_proxy = {}
        mean_fun = {}
        cov1 = {}
        cov2 = {}
        gp = {}

        # GP for each proxy
        for proxy in proxies:

            if ls_dist[proxy] == 'Wald':
                ls_temp = pm.Wald('gp_ls_unshifted_' + proxy, mu = ls_mu[proxy], lam = ls_lambda[proxy], shape = 1)
                gp_ls[proxy] = pm.Deterministic('gp_ls_' + proxy, ls_temp + ls_min[proxy])
            elif ls_dist[proxy] == 'HalfNormal':
                ls_temp = pm.HalfNormal('gp_ls_unshifted_' + proxy, sigma = ls_sigma[proxy], shape = 1)
                gp_ls[proxy] = pm.Deterministic('gp_ls_' + proxy, ls_temp + ls_min[proxy])

            gp_var[proxy] = pm.HalfNormal('gp_var_' + proxy, sigma = var_sigma[proxy], shape=1)

            m_proxy[proxy] = pm.Normal('m_' + proxy, mu = gp_mean_mu[proxy], sigma = gp_mean_sigma[proxy], shape = 1) #

            # mean and covariance functions
            mean_fun[proxy] = pm.gp.mean.Constant(m_proxy[proxy])

            cov1[proxy] = gp_var[proxy] ** 2 * pm.gp.cov.ExpQuad(1, gp_ls[proxy])

            cov2[proxy] = pm.gp.cov.WhiteNoise(white_noise_sigma[proxy])

            # GP prior
            if not approximate:
                gp[proxy] = pm.gp.Latent(mean_func = mean_fun[proxy], cov_func = cov1[proxy] + cov2[proxy])

            if approximate:
                print('Using HSGP approximation for ', proxy)
                gp[proxy] = pm.gp.HSGP(m = [hsgp_m[proxy]], c = hsgp_c[proxy], mean_func = mean_fun[proxy], cov_func = cov1[proxy])

        ages_all = []
        proxy_all = {}
        proxy_sigma_all = {}
        proxy_population_std_all = {}
        include_idx_all = {}

        for proxy in proxies:
            proxy_all[proxy] = []
            include_idx_all[proxy] = []
            proxy_sigma_all[proxy] = []
            proxy_population_std_all[proxy] = []

            if noise_type[proxy] != 'none':
                noise_all[proxy] = []

            if offset_type[proxy] != 'none':
                offset_all[proxy] = []

        # create distribution objects for each unique shared constraint
        # in this implementation, constraints should only be labeled as 'shared?' if diachronous behavior is not allowed -- constraints that are the same (e.g.,
        # a fossil first appearanace date), but that may be diachronous between sections, should be set to shared = False
        shared_constraints = {}
        shared_ages = ages_df[ages_df['shared?']==True]

        if len(shared_ages) > 0:
            unique_shared_constraints = np.unique(shared_ages['name'])

            for constraint in unique_shared_constraints:
                constraint = str(constraint)
                constraint_df = shared_ages[shared_ages['name']==constraint]
                dist = np.unique(constraint_df['distribution_type'])
                dist_age = np.unique(constraint_df['age'])
                dist_age_std = np.unique(constraint_df['age_std'])

                if (len(dist) > 1) or (len(dist_age) > 1) or (len(dist_age_std) > 1):
                    sys.exit(f"Initialization of shared age constraint {constraint} is inconsistent. Check that distribution type and parameters are the same for each section.")

                dist = dist[0]

                if dist == 'Normal':
                    shared_constraints[constraint] = pm.Normal(constraint, mu = dist_age[0], sigma = dist_age_std[0])

                else:
                     # if not implemented, throw error
                    if dist not in DIST_DICT.keys():
                        sys.exit(f"{dist} distribution not implemented. Add to DIST_DICT or choose a different distribution.")

                    dist_args = {}
                    param_1 = constraint_df['param_1'].values[0]
                    param_1_name = constraint_df['param_1_name'].values[0]
                    param_2 = constraint_df['param_2'].values[0]
                    param_2_name = constraint_df['param_2_name'].values[0]

                    if not pd.isna(param_1):
                        dist_args[param_1_name] = param_1

                    if not pd.isna(param_2):
                        dist_args[param_2_name] = param_2

                    shared_constraints[constraint] = DIST_DICT[dist](constraint, **dist_args)

        # if offset and/or noise terms are manually grouped by the user, create distribution for each group
        # separate offset and term for each proxy
        for proxy in proxies:
            if offset_type[proxy] == 'groups':
                offset_group_dict[proxy] = {}

                offset_groups = np.unique(sample_df['offset_group_' + proxy])
                for group in offset_groups:
                    # if using the default laplace prior, and offset_b and offset_mu aren't superseded by parameters passed in offset_params dict
                    if (offset_prior[proxy] == 'Laplace') and (offset_params[proxy] is None):
                        offset_group_dict[proxy][group] = pm.Laplace(group + '_group_offset_' + proxy, mu = offset_mu[proxy], b = offset_b[proxy], shape = 1)

                    else:
                     # if not implemented, throw error
                        if offset_prior[proxy] not in DIST_DICT.keys():
                            sys.exit(f"{offset_prior[proxy]} distribution not implemented. Add to DIST_DICT or choose a different distribution for  `offset_prior`, {proxy}.")

                        dist_args = {}
                        param_1 = offset_params[proxy]['param_1']
                        param_1_name = offset_params[proxy]['param_1_name']
                        param_2 = offset_params[proxy]['param_2']
                        param_2_name = offset_params[proxy]['param_2_name']

                        if not pd.isna(param_1):
                            dist_args[param_1_name] = param_1

                        if not pd.isna(param_2):
                            dist_args[param_2_name] = param_2

                        offset_group_dict[proxy][group] = DIST_DICT[offset_prior[proxy]](group + '_group_offset_' + proxy, **dist_args, shape = 1)

            if noise_type[proxy] == 'groups':
                noise_group_dict[proxy] = {}

                noise_groups = np.unique(sample_df['noise_group_' + proxy])

                for group in noise_groups:
                    if noise_prior[proxy] == 'HalfCauchy':
                        noise_group_dict[proxy][group] = pm.HalfCauchy(group + '_group_noise_' + proxy, beta = noise_beta[proxy], shape = 1)
                    elif noise_prior[proxy] == 'HalfNormal':
                        noise_group_dict[proxy][group] = pm.HalfNormal(group + '_group_noise_' + proxy, sigma = noise_sigma[proxy], shape = 1)
                    elif noise_prior[proxy] == 'HalfStudentT':
                        noise_group_dict[proxy][group] = pm.HalfStudentT(group + '_group_noise_' + proxy, nu = noise_nu[proxy], sigma = noise_sigma[proxy], shape = 1)
                    else:
                        # throw an error if not HalfCauchy, HalfNormal, or HalfStudentT  -- custom noise priors not allowed (since noise prior must be one of the positive-only distributions)
                        sys.exit(f"{noise_prior[proxy]} noise prior not implemented. Options are HalfCauchy (default), HalfNormal, or HalfStudentT.")


        # dictionary to store sample age distributions for each section -- required for superposition between sections
        section_age_dist = {}

        for section in sections:
            intervals = []
            ages = []

            section_df = sample_df[sample_df['section']==section]

            # separate dataframes for intermediate detrital or intrusive constraints
            section_age_df = ages_df[(ages_df['section']==section) & (~ages_df['intermediate detrital?']) & (~ages_df['intermediate intrusive?'])]
            intermediate_detrital_section_ages_df = ages_df[(ages_df['section']==section) & (ages_df['intermediate detrital?'])]
            intermediate_intrusive_section_ages_df =  ages_df[(ages_df['section']==section) & (ages_df['intermediate intrusive?'])]

            section_ages = section_age_df['age'].values
            section_ages_unc = section_age_df['age_std'].values

            heights = section_df['height'].values
            age_heights = section_age_df['height'].values

            # create age constraint distributions
            if len(age_heights) == 0:
                print(str(section) + ' has no age constraints')

            else:
                ages = {}
                label = str(section) +'_'

                # the input ages (the means) need to be in stratigraphic superposition, but ordered transform is still used so that the posteriors cannot be out of superposition due to 2sigma uncertainty -- avoids potentially bad initialization

                # biuld distributions if all age constraint priors are Gaussian and not shared
                if all(section_age_df['distribution_type']=='Normal') and all(section_age_df['shared?']==False):
                    # for initvals, using np.sort instead of np.flip to account for scenario where means of reported ages are out of superposition
                    radiometric_age_flip = pm.Normal(label + 'flip_radiometric_age',
                                                     mu = np.flip(section_ages),
                                                     sigma = np.flip(section_ages_unc),
                                                     shape = section_ages.shape,
                                                     transform = tr.Ordered(),
                                                     initval = np.sort(section_ages))

                    radiometric_age_tensor = pm.Deterministic(label + 'radiometric_age', np.flip(radiometric_age_flip))


                else:
                    print('Using radiometric age priors specified in ages_df for section ' + str(section))
                    radiometric_age = {}
                    age_dist_names = []

                    for i in np.arange(len(section_ages)):
                        label = str(section) + '_' + str(i) + '_'
                        dist = section_age_df['distribution_type'].values[i]

                        constraint_shared = section_age_df['shared?'].values[i]

                        # for shared constraints, link to existing distribution
                        if constraint_shared == True:
                            shared_constraint_name = section_age_df['name'].values[i]
                            radiometric_age[i] = shared_constraints[shared_constraint_name]
                            age_dist_names.append(shared_constraint_name)

                        # if the constraint is not shared, build a new distribution
                        else:
                            if dist == 'Normal':
                                radiometric_age[i] = pm.Normal(label + 'radiometric_age', mu = section_ages[i], sigma = section_ages_unc[i])
                                age_dist_names.append(label  + 'radiometric_age')

                            else:
                                # if not implemented, throw error
                                if dist not in DIST_DICT.keys():
                                    sys.exit(f"{dist} distribution not implemented. Add to DIST_DICT or choose a different distribution.")

                                dist_args = {}
                                param_1 = section_age_df['param_1'].values[i]
                                param_1_name = section_age_df['param_1_name'].values[i]
                                param_2 = section_age_df['param_2'].values[i]
                                param_2_name = section_age_df['param_2_name'].values[i]

                                if not pd.isna(param_1):
                                    dist_args[param_1_name] = param_1

                                if not pd.isna(param_2):
                                    dist_args[param_2_name] = param_2

                                radiometric_age[i] = DIST_DICT[dist](label + 'radiometric_age', **dist_args)

                                age_dist_names.append(label + 'radiometric_age')

                    # make a vector of all the radiometric ages for superposition function
                    radiometric_age_tensor = at.zeros((len(section_ages),))
                    for i in np.arange(len(section_ages)):
                        radiometric_age_tensor = at.set_subtensor(radiometric_age_tensor[i], radiometric_age[i])

                    label = str(section) +'_'
                    radiometric_age_tensor = pm.Deterministic(label + 'radiometric_age', radiometric_age_tensor)

                    superposition(radiometric_age_tensor, age_dist_names, model, section_age_df, section)

                # if there are samples below the basal age constraint, throw an error
                if (heights[0] < age_heights[0]):
                    sys.exit(f"Section {section} does not have a basal age constraint. Add a maximum section age to ages_df.")

                # if the section has no upper age constraint, throw an error
                if heights[-1] >= age_heights[-1]:
                    sys.exit(f"Section {section} does not have an upper age constraint. Add a minimum section age to ages_df.")

                # create sample age distributions for section (by interval)
                for interval in np.arange(0, len(age_heights)-1).tolist():
                    label = str(section)+'_'+ str(interval) +'_'
                    above = section_df['height']>=age_heights[interval]
                    below = section_df['height']<age_heights[interval+1]
                    interval_df = section_df[above & below]
                    interval_samples = interval_df[proxy].values

                    above = intermediate_detrital_section_ages_df['height']>age_heights[interval]
                    below = intermediate_detrital_section_ages_df['height']<age_heights[interval+1]
                    detrital_interval_df = intermediate_detrital_section_ages_df[above & below]

                    above = intermediate_intrusive_section_ages_df['height']>age_heights[interval]
                    below = intermediate_intrusive_section_ages_df['height']<age_heights[interval+1]
                    intrusive_interval_df = intermediate_intrusive_section_ages_df[above & below]

                    # if there are samples in the current interval
                    # if interval = 0, and the current section is in the superposition dictionary, then use the minimum age from sections that must be older as the maximum age for the current section
                    # note: if appropriate, make sure that the overlying age constraint is shared between the sections to avoid potential superposition issues
                    if len(interval_samples) > 0:

                        if section in list(superposition_dict.keys()):
                            base_age_dist = pm.math.concatenate([section_age_dist[older_section] for older_section in superposition_dict[section]]).min()
                        else:
                            base_age_dist = radiometric_age_tensor[interval]

                        observed_age_diff = pm.Deterministic(label + 'obs_age_diff', base_age_dist - radiometric_age_tensor[interval+1])

                        # sort random draws if >1 sample
                        if len(interval_samples) > 1:

                            random_sample_ages_unsorted = pm.Uniform(label + 'unsorted_random_ages', lower = 0, upper = 1, size = len(interval_samples))

                            random_sample_ages = pm.Deterministic(label + 'random_ages', at.sort(random_sample_ages_unsorted))

                        # skip sorting if interval only contains 1 sample
                        else:
                            random_sample_ages =  pm.Uniform(label + 'random_ages', lower = 0, upper = 1, size = len(interval_samples))

                        # scaled age parameterization
                        scaling_factor_1 = pm.Uniform(label + 'scaling_factor_1', lower = 0, upper = 1, size = 1)
                        scaling_factor_2 = pm.Uniform(label + 'scaling_factor_2', lower = 0, upper = 1, size = 1)

                        # scaled_ages = max - random_ages * age_range * sf1 - (1 - sf2) * age_range
                        ages[interval] = pm.Deterministic(label + 'ages', base_age_dist - (random_sample_ages * observed_age_diff * scaling_factor_1) - (1 - scaling_factor_2) * observed_age_diff * (1 - scaling_factor_1))

                        intervals.append(interval)


                        # if there are intermediate detrital ages in the section, check if they're inside the current interval (iterate over constraints)
                        if detrital_interval_df.shape[0] > 0:
                            for i in np.arange(detrital_interval_df.shape[0]):
                                # if there are overlying samples in interval, enforce maximum age
                                if len(interval_df[interval_df['height']>=detrital_interval_df['height'].values[i]]['height'].values)>0:
                                    # construct DZ age prior
                                    dist = detrital_interval_df['distribution_type'].values[i]
                                    constraint_shared = detrital_interval_df['shared?'].values[i]

                                    if constraint_shared == True:
                                        shared_constraint_name = detrital_interval_df['name'].values[i]
                                        intermediate_detrital_age = shared_constraints[shared_constraint_name]
                                        intermediate_detrital_age_dist_name = shared_constraint_name

                                    else:
                                        if dist == 'Normal':
                                            intermediate_detrital_age = pm.Normal(label + 'detrital_age_' + str(i),
                                                                           mu = detrital_interval_df['age'].values[i],
                                                                           sigma = detrital_interval_df['age_std'].values[i])
                                            intermediate_detrital_age_dist_name = label + 'detrital_age_' + str(i)

                                        else:
                                            # if distribution not implemented, throw error
                                            if dist not in DIST_DICT.keys():
                                                sys.exit(f"{dist} distribution not implemented. Add to DIST_DICT or choose a different distribution.")

                                            dist_args = {}
                                            param_1 = detrital_interval_df['param_1'].values[i]
                                            param_1_name = detrital_interval_df['param_1_name'].values[i]
                                            param_2 = detrital_interval_df['param_2'].values[i]
                                            param_2_name = detrital_interval_df['param_2_name'].values[i]

                                            if not pd.isna(param_1):
                                                dist_args[param_1_name] = param_1

                                            if not pd.isna(param_2):
                                                dist_args[param_2_name] = param_2

                                            intermediate_detrital_age = DIST_DICT[dist](label + 'detrital_age_' + str(i), **dist_args)

                                            intermediate_detrital_age_dist_name = label + 'detrital_age_' + str(i)

                                    # enforce detrital age with pm.Potential
                                    if all(section_age_df['distribution_type']=='Normal') and all(section_age_df['shared?']==False):
                                        # THESE WILL BE UPSIDE DOWN -- inside of the DZ potential, use this name, but flip the initial values
                                        # with np.flip() THEN index with [interval] and [interval+1]
                                        base_age_dist_name = str(section) +'_' + 'flip_radiometric_age'
                                        upper_age_dist_name = str(section) +'_' + 'flip_radiometric_age'
                                        shared_radiometric_age_dist = True

                                    else:
                                        base_age_dist_name = age_dist_names[interval]
                                        upper_age_dist_name = age_dist_names[interval + 1]
                                        shared_radiometric_age_dist = False

                                    intermediate_detrital_potential(intermediate_detrital_age,
                                                                    intermediate_detrital_age_dist_name,
                                                                    base_age_dist_name,
                                                                    upper_age_dist_name,
                                                                    ages[interval],
                                                                    random_sample_ages_unsorted,
                                                                    label + 'unsorted_random_ages',
                                                                    interval_df['height'].values,
                                                                    detrital_interval_df['height'].values[i],
                                                                    model,
                                                                    section,
                                                                    interval,
                                                                    sf1_name = label + 'scaling_factor_1',
                                                                    sf2_name = label + 'scaling_factor_2',
                                                                    shared_radiometric_age_dist = shared_radiometric_age_dist
                                                                    )

                                    # variables:
                                    # dz age: intermediate_detrital_age
                                    # dz age name: intermediate_detrital_age_dist_name
                                    # sample ages (unsorted random draws U[0, 1]): random_sample_ages_unsorted
                                    # sample age dist name: label + 'unsorted_random_ages'
                                    # base age dist name: see above
                                    # upper age dist name: see above
                                    # scaling factor 1 name: label + 'scaling_factor_1'
                                    # scaling factor 2 name: label + 'scaling_factor_2'

                        # if there are intermediate intrusive ages in section, check if they're inside this interval
                        if intrusive_interval_df.shape[0] > 0:
                            for i in np.arange(intrusive_interval_df.shape[0]):
                                # if there are underlying samples in interval, enforce maximum age
                                if len(interval_df[interval_df['height']<=intrusive_interval_df['height'].values[i]]['height'].values)>0:
                                    # construct intrusive age prior
                                    dist = intrusive_interval_df['distribution_type'].values[i]
                                    constraint_shared = intrusive_interval_df['shared?'].values[i]

                                    if constraint_shared == True:
                                        shared_constraint_name = intrusive_interval_df['name'].values[i]
                                        intermediate_intrusive_age = shared_constraints[shared_constraint_name]
                                        intermediate_intrusive_age_dist_name = shared_constraint_name

                                    else:
                                        if dist == 'Normal':
                                            intermediate_intrusive_age = pm.Normal(label + 'intrusive_age_' + str(i),
                                                                           mu = intrusive_interval_df['age'].values[i],
                                                                           sigma = intrusive_interval_df['age_std'].values[i])
                                            intermediate_intrusive_age_dist_name = label + 'intrusive_age_' + str(i)

                                        else:
                                            # if distribution not implemented, throw error
                                            if dist not in DIST_DICT.keys():
                                                sys.exit(f"{dist} distribution not implemented. Add to DIST_DICT or choose a different distribution.")

                                            dist_args = {}
                                            param_1 = intrusive_interval_df['param_1'].values[i]
                                            param_1_name = intrusive_interval_df['param_1_name'].values[i]
                                            param_2 = intrusive_interval_df['param_2'].values[i]
                                            param_2_name = intrusive_interval_df['param_2_name'].values[i]

                                            if not pd.isna(param_1):
                                                dist_args[param_1_name] = param_1

                                            if not pd.isna(param_2):
                                                dist_args[param_2_name] = param_2

                                            intermediate_intrusive_age = DIST_DICT[dist](label + 'intrusive_age_' + str(i), **dist_args)

                                            intermediate_intrusive_age_dist_name = label + 'intrusive_age_' + str(i)

                                    # enforce intermediate age with pm.Potential
                                    if all(section_age_df['distribution_type']=='Normal') and all(section_age_df['shared?']==False):
                                        # THESE WILL BE UPSIDE DOWN -- inside of the intrusive potential, use this name, but flip the initial values
                                        # with np.flip() THEN index with [interval] and [interval+1]
                                        base_age_dist_name = str(section) +'_' + 'flip_radiometric_age'
                                        upper_age_dist_name = str(section) +'_' + 'flip_radiometric_age'
                                        shared_radiometric_age_dist = True

                                    else:
                                        base_age_dist_name = age_dist_names[interval]
                                        upper_age_dist_name = age_dist_names[interval + 1]
                                        shared_radiometric_age_dist = False


                                    intermediate_intrusive_potential(intermediate_intrusive_age,
                                                                    intermediate_intrusive_age_dist_name,
                                                                    base_age_dist_name,
                                                                    upper_age_dist_name,
                                                                    ages[interval],
                                                                    random_sample_ages_unsorted,
                                                                    label + 'unsorted_random_ages',
                                                                    interval_df['height'].values,
                                                                    intrusive_interval_df['height'].values[i],
                                                                    model,
                                                                    section,
                                                                    interval,
                                                                    sf1_name = label + 'scaling_factor_1',
                                                                    sf2_name = label + 'scaling_factor_2',
                                                                    shared_radiometric_age_dist = shared_radiometric_age_dist
                                                                    )


                label = str(section) + '_'

                # concatenate ages from all intervals in section
                ages = [ages[interval] for interval in intervals]

                section_age_tensor = at.zeros((len(heights),))

                count = 0
                for age_sub in ages:
                    for i in np.arange(0, age_sub.shape.eval()[0]):
                        section_age_tensor = at.set_subtensor(section_age_tensor[count], age_sub[i])
                        count += 1

                section_age_dist[section] = pm.Deterministic(label+'ages', section_age_tensor)

                ages_all = np.append(ages_all, ages)

                proxy_obs = {}
                include_idx = {}
                proxy_sigma_obs = {}
                proxy_population_std_obs = {}

                if ('section' in noise_types) or ('groups' in noise_types):
                    section_noise = {}

                if ('section' in offset_types) or  ('groups' in offset_types):
                    section_offset = {}

                # set up the noise and offset terms, and update the proxy observation lists
                for proxy in proxies:
                    # ignore samples marked 'exclude' + nans (used for indexing later -- for now, concatenate all values)
                    include_idx[proxy] = (~section_df['Exclude?'].values.astype(bool)) & (~np.isnan(section_df[proxy]))
                    proxy_obs[proxy] = section_df[proxy].values
                    proxy_sigma_obs[proxy] = section_df[proxy + '_std'].values
                    proxy_population_std_obs[proxy] = section_df[proxy + '_population_std'].values

                    if offset_type[proxy] == 'section':
                        # if using the default Laplace prior, and offset_b and offset_mu aren't superseded by parameters passed in offset_params dict
                        if (offset_prior[proxy] == 'Laplace') and (offset_params[proxy] is None):
                            section_offset[proxy] = pm.Laplace(label + 'section_offset_' + proxy, mu = offset_mu[proxy], b = offset_b[proxy], shape = 1)

                        else:
                            # if requested offset distribution not implemented, throw error
                            if offset_prior[proxy] not in DIST_DICT.keys():
                                sys.exit(f"{offset_prior[proxy]} distribution not implemented. Add to DIST_DICT or choose a different distribution for  `offset_prior`, {proxy}.")

                            dist_args = {}
                            param_1 = offset_params[proxy]['param_1']
                            param_1_name = offset_params[proxy]['param_1_name']
                            param_2 = offset_params[proxy]['param_2']
                            param_2_name = offset_params[proxy]['param_2_name']

                            if not pd.isna(param_1):
                                dist_args[param_1_name] = param_1

                            if not pd.isna(param_2):
                                dist_args[param_2_name] = param_2

                            section_offset[proxy] = DIST_DICT[offset_prior[proxy]](label + 'section_offset_' + proxy, **dist_args, shape = 1)

                        # only create offset distributions for samples that will be included in the likelihood function
                        offset_all[proxy].append(([1] * len(heights[include_idx[proxy]])) * section_offset[proxy])

                    elif offset_type[proxy] == 'groups':
                        section_offset[proxy] = []
                        offset_group_list = section_df['offset_group_' + proxy][include_idx[proxy]].values
                        for offset_key in offset_group_list:
                            section_offset[proxy].append(offset_group_dict[proxy][offset_key])

                        offset_all[proxy].append(section_offset[proxy])

                    if noise_type[proxy] == 'section':
                        if noise_prior[proxy] =='HalfCauchy':
                            section_noise[proxy] = pm.HalfCauchy(label + 'section_noise_' + proxy, beta = noise_beta[proxy], shape = 1) # beta = 2
                        elif noise_prior[proxy] == 'HalfNormal':
                            section_noise[proxy] = pm.HalfNormal(label + 'section_noise_' + proxy, sigma = noise_sigma[proxy], shape = 1)
                        elif noise_prior[proxy] == 'HalfStudentT':
                            section_noise[proxy] = pm.HalfStudentT(label + 'section_noise_' + proxy, nu = noise_nu[proxy], sigma = noise_sigma[proxy], shape = 1)

                        noise_all[proxy].append(([1] * len(heights[include_idx[proxy]])) * section_noise[proxy])

                    elif noise_type[proxy] == 'groups':
                        section_noise[proxy] = []
                        noise_group_list = section_df['noise_group_' + proxy][include_idx[proxy]].values
                        for noise_key in noise_group_list:
                            section_noise[proxy].append(noise_group_dict[proxy][noise_key])

                        noise_all[proxy].append(section_noise[proxy])

                    proxy_all[proxy] = np.append(proxy_all[proxy], proxy_obs[proxy])
                    include_idx_all[proxy] = np.append(include_idx_all[proxy], include_idx[proxy])
                    proxy_sigma_all[proxy] = np.append(proxy_sigma_all[proxy], proxy_sigma_obs[proxy])
                    proxy_population_std_all[proxy] = np.append(proxy_population_std_all[proxy], proxy_population_std_obs[proxy])

        age_tensor = at.zeros((len(proxy_all[proxies[0]]),))

        count = 0
        for sub in np.arange(0, len(ages_all)):
            for i in np.arange(0, ages_all[sub].shape.eval()[0]):
                age_tensor = at.set_subtensor(age_tensor[count], ages_all[sub][i])
                count += 1

        # shared sample ages
        ages = pm.Deterministic('ages', age_tensor)

        # likelihood function for each proxy
        proxy_idx = {}
        proxy_sigma_vec = {}
        proxy_population_std_vec = {}
        proxy_quadrature_uncertainty = {}

        if ('section' in offset_types) or ('groups' in offset_types):
            offset = {}
            offset_tensor = {}

        if ('section' in noise_types) or  ('groups' in noise_types):
            noise = {}
            noise_tensor = {}

        for proxy in proxies:
            # convert boolean vector (ignore 'exclude' and nans) to list of indices
            proxy_idx[proxy] = np.where(include_idx_all[proxy])[0] #np.where(~np.isnan(proxy_all[proxy]))[0]
            proxy_sigma_vec[proxy] = proxy_sigma_all[proxy][proxy_idx[proxy]]
            proxy_sigma_vec[proxy][np.isnan(proxy_sigma_vec[proxy])] = proxy_sigma_default[proxy]

            proxy_population_std_vec[proxy] = proxy_population_std_all[proxy][proxy_idx[proxy]]
            proxy_population_std_vec[proxy][np.isnan(proxy_population_std_vec[proxy])] = 0

            proxy_quadrature_uncertainty[proxy] = np.sqrt((proxy_sigma_vec[proxy])**2 + (proxy_population_std_vec[proxy])**2)

            # construct final offset terms
            if offset_type[proxy] != 'none':
                offset_tensor[proxy] =  at.zeros((len(proxy_all[proxy][proxy_idx[proxy]]),))
                count = 0

                # custom offset groups
                if offset_type[proxy] == 'groups':
                    for sub in np.arange(0, len(offset_all[proxy])):
                        for i in np.arange(0, len(offset_all[proxy][sub])):
                            offset_tensor[proxy] = at.set_subtensor(offset_tensor[proxy][count], offset_all[proxy][sub][i][0])
                            count +=1
                    offset[proxy] = pm.Deterministic('offset_' + proxy, offset_tensor[proxy])

                elif offset_type[proxy] == 'section':
                    for sub in np.arange(0, len(offset_all[proxy])):
                        for i in np.arange(0, offset_all[proxy][sub].shape.eval()[0]):
                            offset_tensor[proxy] = at.set_subtensor(offset_tensor[proxy][count], offset_all[proxy][sub][i])
                            count +=1
                    offset[proxy] = pm.Deterministic('offset_' + proxy, offset_tensor[proxy])

            # construct final noise terms
            if noise_type[proxy] != 'none':
                noise_tensor[proxy] =  at.zeros((len(proxy_all[proxy][proxy_idx[proxy]]),))
                count = 0
                # custom noise groups
                if noise_type[proxy] == 'groups':
                    for sub in np.arange(0, len(noise_all[proxy])):
                        for i in np.arange(0, len(noise_all[proxy][sub])):
                            noise_tensor[proxy] = at.set_subtensor(noise_tensor[proxy][count], noise_all[proxy][sub][i][0])
                            count +=1
                    noise[proxy] = pm.Deterministic('noise_' + proxy, noise_tensor[proxy])

                # per-section noise
                elif noise_type[proxy] == 'section':
                    for sub in np.arange(0, len(noise_all[proxy])):
                        for i in np.arange(0, noise_all[proxy][sub].shape.eval()[0]):
                            noise_tensor[proxy] = at.set_subtensor(noise_tensor[proxy][count], noise_all[proxy][sub][i])
                            count +=1
                    noise[proxy] = pm.Deterministic('noise_' + proxy, noise_tensor[proxy])

            if not approximate:
                f = gp[proxy].prior('f_' + proxy, X=ages[proxy_idx[proxy],None],
                             reparameterize=True,
                             jitter = jitter)

            if approximate:
                f = gp[proxy].prior('f_' + proxy, X=ages[proxy_idx[proxy],None])

            if proxy_observed:
                if offset_type[proxy] != 'none':
                    proxy_pred = pm.Normal(proxy + '_pred', mu=f.flatten() + offset[proxy],
                                        sigma = proxy_quadrature_uncertainty[proxy] + noise[proxy],
                                        shape = proxy_all[proxy][proxy_idx[proxy]].shape,
                                        observed=proxy_all[proxy][proxy_idx[proxy]])


                else:
                    proxy_pred = pm.Normal(proxy + '_pred', mu=f.flatten(),
                                    sigma = proxy_quadrature_uncertainty[proxy] + noise[proxy],
                                    shape = proxy_all[proxy][proxy_idx[proxy]].shape,
                                    observed=proxy_all[proxy][proxy_idx[proxy]])

            else:
                if offset_type[proxy] != 'none':
                    proxy_pred = pm.Normal(proxy + '_pred', mu=f.flatten() + offset[proxy],
                                    sigma = proxy_quadrature_uncertainty[proxy] + noise[proxy],
                                    shape = proxy_all[proxy][proxy_idx[proxy]].shape)


                else:
                    proxy_pred = pm.Normal(proxy + '_pred', mu=f.flatten(),
                                    sigma = proxy_quadrature_uncertainty[proxy] + noise[proxy],
                                    shape = proxy_all[proxy][proxy_idx[proxy]].shape)

    return model, gp



def superposition(age_dist, age_dist_names, model, section_age_df, section):
    """
    Helper function for explicitly enforcing stratigraphic superposition (any given sample must be younger than the underlying sample) for a group of radiometric age constraints.

    Parameters
    ----------
    age_dist: pymc.distribtions
        :class:`pymc.distributions` object for the radiometric ages (must be in stratigraphic order).

    age_dist_name: str
        Name of ``age_dist`` in ``model``.

    model: pymc.Model
        :class:`pymc.model.core.Model` object associated with the distributions in ``age_dist``.

    section_age_df: pandas.DataFrame
        :class:`pandas.DataFrame` containing age constraints for ``section``.

    section: str
        Name of the section containing the radiometric age constraints.

    """

    for i in np.arange(1, age_dist.shape.eval()[0]):
        slope = 100
        pm.Potential("superposition_"+str(section)+'_'+str(i),
        shared(-10000) * pm.math.invlogit(slope * ((age_dist[i]) - age_dist[i-1])))

    for i in np.arange(0, len(section_age_df['height'])-1).tolist():
        lower_label = age_dist_names[i]
        upper_label = age_dist_names[i + 1]

        # note that model.initial_point returns transformed values, while model initial values are untransformed (?)
        # so we get the initial points, and then have to apply the backward transform to 'convert' them to initial values for the untransformed distributions and check if they are in superposition
        rv_var_base = model[lower_label]

        base_age_initval = transformed_initval(lower_label,  model)

        rv_var_upper = model[upper_label]

        upper_age_initval = transformed_initval(upper_label, model)

        age_diff = base_age_initval - upper_age_initval

        # if model is starting out of superposition, make upper sample younger by 0.1; otherwise, leave as-is:
        if age_diff > 0:
            model.set_initval(rv_var_base, base_age_initval)
            model.set_initval(rv_var_upper, upper_age_initval)
        else:
            model.set_initval(rv_var_base, base_age_initval)
            model.set_initval(rv_var_upper, base_age_initval - shared(0.2))

def intermediate_detrital_potential(detrital_age_dist, detrital_age_dist_name, maximum_age_dist_name, minimum_age_dist_name, sample_age_dist_sorted, sample_age_dist, sample_age_dist_name, sample_heights, detrital_height, model, section, interval, sf1_name, sf2_name, shared_radiometric_age_dist = True):
    """
    Helper function for enforcing detrital age constraints (maximum age constraint for all overlying samples; no constraint on underlyig samples) in the middle of a section.

    Parameters
    ----------
    detrital_age_dist: pymc.distributions
        Detrital age distribution.

    detrital_age_dist_name: str
        Name of ``detrital_age_dist`` in ``model``.

    maximum_age_dist_name: str
        Name of distribution for underlying maximum age constraint in ``model``.

    minimum_age_dist_name: str
        Name of distribution for overlying minimum age constraint in ``model``.

    sample_age_dist_sorted: pymc.distribtions
        Sorted and scaled sample age distribitions.

    sample_age_dist: pymc.distributions
        Unsorted and unscaled sample age distributions.

    sample_age_dist_name: str
        Name of ``sample_age_dist`` in the pymc.Model object.

    sample_heights: np.array
        Array of heights for samples in the current interval.

    detrital_height: float
        Height of detrital age constraint.

    model: pymc.Model
        :class:`pymc.model.core.Model` object associated with the input distributions.

    section: str
        Name of the current section.

    interval: int
        Current interval number.

    sf1_name: str
        Name of the distribution associated with scaling factor 1 in ``model``.

    sf2_name: str
        Name of the distribution associated with scaling factor 2 in ``model``.

    shared_radiometric_age_dist: bool, optional
        Whether the radiometric age distributions are part of a single object (versus initiated as separate distributions). Defaults to ``True``.

    """

    overlying_sample_idx = np.where(sample_heights >= detrital_height)[0]

    slope = 100
    pm.Potential('detrital_max_'  + str(section) + '_' + str(detrital_age_dist_name), shared(-10000) * pm.math.invlogit(slope * ((sample_age_dist_sorted[overlying_sample_idx]) - detrital_age_dist)))

    # get initial values for sample ages
    rv_var_samples = model[sample_age_dist_name]

    sample_age_initval = transformed_initval(sample_age_dist_name, model)

    # get initial value for detrital age constraint
    dz_age_initval = transformed_initval(detrital_age_dist_name, model)

    # get initial value for underlying minimum age constraint
    base_age_initval = transformed_initval(maximum_age_dist_name, model)

    if shared_radiometric_age_dist:
        if len(base_age_initval) > 1:
            base_age_initval = np.flip(base_age_initval)[interval]

    # get initial value for overlying minimum age constraint
    upper_age_initval = transformed_initval(minimum_age_dist_name, model)

    if shared_radiometric_age_dist:
        if len(upper_age_initval) > 1:
            upper_age_initval = np.flip(upper_age_initval)[interval+1]


    # set initial values for sample ages
    # get initial values for scaling factors
    rv_var_sf1 = model[sf1_name]

    sf1_initval = transformed_initval(sf1_name, model)

    rv_var_sf2 = model[sf2_name]

    sf2_initval = transformed_initval(sf2_name, model)

    # calculate the bounds of the current 'age box', given sf1 and sf2
    current_max = base_age_initval - (1 - sf2_initval) * (base_age_initval - upper_age_initval) * (1 - sf1_initval)
    current_min = base_age_initval - ((base_age_initval - upper_age_initval) * sf1_initval) - (1 - sf2_initval) * (base_age_initval - upper_age_initval) * (1 - sf1_initval)

    # calculate DZ age on [0, 1] scale defined by bounds
    scaled_dz_initval = (current_max - dz_age_initval)/(current_max - current_min)

    # if ages younger than the DZ are outside of the current bounds:
    # reset sf1 and sf2 initival values until the DZ age can be enforced, update sample age initvals
    if scaled_dz_initval > 1:
        while scaled_dz_initval > 1:
            sf1_initval = np.random.uniform(0, 1, 1)
            sf2_initval = np.random.uniform(0, 1, 1)

            current_max = base_age_initval - (1 - sf2_initval) * (base_age_initval - upper_age_initval) * (1 - sf1_initval)
            current_min = base_age_initval - ((base_age_initval - upper_age_initval) * sf1_initval) - (1 - sf2_initval) * (base_age_initval - upper_age_initval) * (1 - sf1_initval)

            scaled_dz_initval = (current_max - dz_age_initval)/(current_max - current_min)

        scaled_age_initial_points = np.random.uniform(scaled_dz_initval, 1, len(overlying_sample_idx))

        sample_age_initval[overlying_sample_idx] = scaled_age_initial_points

        model.set_initval(rv_var_sf1, sf1_initval)
        model.set_initval(rv_var_sf2, sf2_initval)
        model.set_initval(rv_var_samples, sample_age_initval)

    # if the scaled DZ is in range, but the sample age initvals don't respect the constraint:
    # keep sf1/sf2 initvals, but update sample age initvals
    elif (scaled_dz_initval < 1) & (len(sample_age_initval[sample_age_initval >= scaled_dz_initval]) < len(overlying_sample_idx)):

        scaled_age_initial_points = np.random.uniform(scaled_dz_initval, 1, len(overlying_sample_idx))

        sample_age_initval[overlying_sample_idx] = scaled_age_initial_points

        model.set_initval(rv_var_samples, sample_age_initval)



def intermediate_intrusive_potential(intrusive_age_dist, intrusive_age_dist_name, maximum_age_dist_name, minimum_age_dist_name, sample_age_dist_sorted, sample_age_dist, sample_age_dist_name, sample_heights, intrusive_height, model, section, interval, sf1_name, sf2_name, shared_radiometric_age_dist = True, **kwargs):
    """
    Helper function for enforcing intrusive age constraints (minimum age constraint for all underlying samples; no constraint on overlying samples) in the middle of a section.

    Parameters
    ----------
    intrusive_age_dist: pymc.distributions
        Intrusive age distribution.

    intrusive_age_dist_name: str
        Name of ``intrusive_age_dist`` in ``model``.

    maximum_age_dist_name: str
        Name of distribution for underlying maximum age constraint in ``model``.

    minimum_age_dist_name: str
        Name of distribution for overlying minimum age constraint in ``model``.

    sample_age_dist_sorted: pymc.distribtions
        Sorted and scaled sample age distribitions.

    sample_age_dist: pymc.distributions
        Unsorted and unscaled sample age distributions.

    sample_age_dist_name: str
        Name of ``sample_age_dist`` in the pymc.Model object.

    sample_heights: np.array
        Array of heights for samples in the current interval.

    intrusive_height: float
        Height of intrusive age constraint.

    model: pymc.Model
        :class:`pymc.model.core.Model` object associated with the input distributions.

    section: str
        Name of the current section.

    interval: int
        Current interval number.

    sf1_name: str
        Name of the distribution associated with scaling factor 1 in ``model``.

    sf2_name: str
        Name of the distribution associated with scaling factor 2 in ``model``.

    shared_radiometric_age_dist: bool, optional
        Whether the radiometric age distributions are part of a single object (versus initiated as separate distributions). Defaults to ``True``.

    """

    underlying_sample_idx = np.where(sample_heights <= intrusive_height)[0]

    slope = 100
    pm.Potential('intrusive_min_'  + str(section) + '_' + str(intrusive_age_dist_name), shared(-10000) * pm.math.invlogit(slope * ((intrusive_age_dist) - sample_age_dist_sorted[underlying_sample_idx])))

    # get initial values for sample ages
    rv_var_samples = model[sample_age_dist_name]

    sample_age_initval = transformed_initval(sample_age_dist_name, model)

    # get initial value for intrusive age constraint
    intrusive_age_initval = transformed_initval(intrusive_age_dist_name, model)

    # get initial value for underlying minimum age constraint
    base_age_initval = transformed_initval(maximum_age_dist_name, model)

    if shared_radiometric_age_dist:
        if len(base_age_initval) > 1:
            base_age_initval = np.flip(base_age_initval)[interval]

    # get initial value for overlying minimum age constraint
    upper_age_initval = transformed_initval(minimum_age_dist_name, model)

    if shared_radiometric_age_dist:
        if len(upper_age_initval) > 1:
            upper_age_initval = np.flip(upper_age_initval)[interval+1]

    # set initial values for sample ages
    # get initial values for scaling factors
    rv_var_sf1 = model[sf1_name]

    sf1_initval = transformed_initval(sf1_name, model)

    rv_var_sf2 = model[sf2_name]

    sf2_initval = transformed_initval(sf2_name, model)

    # calculate the bounds of the current 'age box', given sf1 and sf2
    current_max = base_age_initval - (1 - sf2_initval) * (base_age_initval - upper_age_initval) * (1 - sf1_initval)
    current_min = base_age_initval - ((base_age_initval - upper_age_initval) * sf1_initval) - (1 - sf2_initval) * (base_age_initval - upper_age_initval) * (1 - sf1_initval)

    # calculate intrusive age on [0, 1] scale defined by bounds
    scaled_intrusive_initval = (current_max - intrusive_age_initval)/(current_max - current_min)

    # if ages older than the intrusive age are outside of the current bounds:
    # reset sf1 and sf2 initival values until the intrusive age can be enforced, update sample age initvals
    if scaled_intrusive_initval < 0:
        while scaled_intrusive_initval < 0:
            sf1_initval = np.random.uniform(0, 1, 1)
            sf2_initval = np.random.uniform(0, 1, 1)

            current_max = base_age_initval - (1 - sf2_initval) * (base_age_initval - upper_age_initval) * (1 - sf1_initval)
            current_min = base_age_initval - ((base_age_initval - upper_age_initval) * sf1_initval) - (1 - sf2_initval) * (base_age_initval - upper_age_initval) * (1 - sf1_initval)

            scaled_intrusive_initval = (current_max - intrusive_age_initval)/(current_max - current_min)

        # if scaled initival value is <1, use it to truncate the age box
        # scaled initial value could be >1, which means all the values in the age box already respect the constraint --> do nothing (initial values will already respect constraint)
        if scaled_intrusive_initval < 1:
            scaled_age_initial_points = np.random.uniform(0, scaled_intrusive_initval, len(underlying_sample_idx))

        sample_age_initval[underlying_sample_idx] = scaled_age_initial_points

        model.set_initval(rv_var_sf1, sf1_initval)
        model.set_initval(rv_var_sf2, sf2_initval)
        model.set_initval(rv_var_samples, sample_age_initval)

    # if the scaled intrusive is in range, but the sample age initvals don't respect the constraint:
    # keep sf1/sf2 initvals, but update sample age initvals
    elif (scaled_intrusive_initval > 0) & (len(sample_age_initval[sample_age_initval <= scaled_intrusive_initval]) < len(underlying_sample_idx)):

        # scaled_intrusive_initval should always be >0 and <1 if the above condition is true
        scaled_age_initial_points =np.random.uniform(0, scaled_intrusive_initval, len(underlying_sample_idx))

        sample_age_initval[underlying_sample_idx] = scaled_age_initial_points

        model.set_initval(rv_var_samples, sample_age_initval)



def transformed_initval(var_name, model):

    """
    Helper function to retrieve the transformed initial values for a random variable in a :class:`pymc.model.core.Model` object.

    Parameters
    ----------
    var_name: str
        Name of variable to be transformed (as specified in ``model``).

    model: PyMC model
        :class:`pymc.model.core.Model` object that contains the target random variable.


    """

    rv = model[var_name]

    init = model.initial_point()[
                    model.rvs_to_values[rv].name
                ]

    transform = model.rvs_to_transforms[rv]

    if transform is not None:
        initval = transform.backward(
            init, * rv.owner.inputs
        )
        initval = initval.eval()
    else:
        initval = init

    return initval
