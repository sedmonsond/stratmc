.. toctree::
  :hidden:
  :maxdepth: 2

Quick start guide
===================

Here, we provide a basic example of importing data and running the inference model. More complex examples are available on the :doc:`example notebooks <examples>` page.

1. Fill out ``proxy data`` and ``age constraint`` tables according to the :doc:`data formatting <datatable>` specifications.

2. Pre-process your data for model construction with :py:meth:`load_data() <stratmc.data.load_data>`

    .. code-block:: python

      from stratmc.data import load_data

      sample_data, age_data = load_data('path_to_proxy_data.csv', 'path_to_age_data.csv')

3. Build a :class:`pymc.model.core.Model` with :py:meth:`build_model(sample_data, age_data, proxies = ['proxy']) <stratmc.model.build_model>`

    .. code-block:: python

      from stratmc.model import build_model

      model, gp = build_model(sample_data, age_data, proxies = ['d13c'])

4. Sample the model posterior using a JAX-assisted MCMC sampling algorithm with :py:meth:`get_trace(model, gp, ages) <stratmc.inference.build_model>`.

    .. code-block:: python

      import pymc as pm
      from stratmc.inference import get_trace

      # array of ages at which to sample the posterior proxy curve
      predict_ages = np.linspace(lower_age, upper_age, number_ages)

      trace = get_trace(model, gp, predict_ages)

5. Plot and analyze the results with the :py:mod:`stratmc.plotting` library.

    .. code-block:: python

      from stratmc import plotting

      plotting.proxy_inference(sample_data, age_data, trace)

    .. plot::

      from stratmc.config import PROJECT_ROOT
      from stratmc.data import load_object, load_trace
      from stratmc.plotting import proxy_inference
      full_trace = load_trace('examples/example_docs_trace')
      example_sample_path = 'examples/example_sample_df'
      example_ages_path = 'examples/example_ages_df'
      sample_df = load_object(example_sample_path)
      ages_df = load_object(example_ages_path)
      proxy_inference(sample_df, ages_df, full_trace, proxy = 'd13c')
      plt.show()
