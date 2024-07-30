from stratmc.config import PROJECT_ROOT
from stratmc.data import load_trace
from stratmc.plotting import lengthscale_traceplot

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_trace_convergence')

lengthscale_traceplot(full_trace, chains = [0, 1, 2, 3])

plt.show()