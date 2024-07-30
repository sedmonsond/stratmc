from stratmc.config import PROJECT_ROOT
from stratmc.data import load_trace
from stratmc.plotting import lengthscale_stability

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_trace_convergence')

lengthscale_stability(full_trace)

plt.show()