from stratmc.config import PROJECT_ROOT
from stratmc.data import load_object
from stratmc.plotting import proxy_strat

example_sample_path = str(PROJECT_ROOT) + '/examples/example_sample_df'
example_ages_path = str(PROJECT_ROOT) + '/examples/example_ages_df'
sample_df = load_object(example_sample_path)
ages_df = load_object(example_ages_path)

proxy_strat(sample_df, ages_df, proxy = 'd13c')

plt.show()