from stratmc.config import PROJECT_ROOT
from stratmc.data import load_object, load_trace
from stratmc.plotting import section_age_range

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_trace')
example_sample_path = str(PROJECT_ROOT) + '/examples/example_sample_df'
example_ages_path = str(PROJECT_ROOT) + '/examples/example_ages_df'
sample_df = load_object(example_sample_path)
ages_df = load_object(example_ages_path)

section_age_range(full_trace, sample_df, ages_df, 120, 130)

plt.show()