from stratmc.config import PROJECT_ROOT
from stratmc.data import load_trace
from stratmc.plotting import proxy_data_gaps

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_trace_data_density')

proxy_data_gaps(full_trace)

plt.show()