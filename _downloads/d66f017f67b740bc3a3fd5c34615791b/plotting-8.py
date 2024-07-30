from stratmc.config import PROJECT_ROOT
from stratmc.data import load_object, load_trace
from stratmc.plotting import limiting_age_constraints

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_limiting_ages_trace')
example_sample_path = str(PROJECT_ROOT) + '/examples/example_sample_df_limiting_ages'
example_ages_path = str(PROJECT_ROOT) + '/examples/example_ages_df_limiting_ages'
sample_df = load_object(example_sample_path)
ages_df = load_object(example_ages_path)
section = '1'

limiting_age_constraints(full_trace, sample_df, ages_df, section)

plt.show()