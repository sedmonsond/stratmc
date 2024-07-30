from stratmc.config import PROJECT_ROOT
from stratmc.data import load_object, load_trace
from stratmc.plotting import sample_ages

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_trace')
example_sample_path = str(PROJECT_ROOT) + '/examples/example_sample_df'
sample_df = load_object(example_sample_path)
section = '1'

sample_ages(full_trace, sample_df, section)

plt.show()