from stratmc.config import PROJECT_ROOT
from stratmc.data import load_trace
from stratmc.plotting import proxy_signal_stability

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_trace_convergence')

proxy_signal_stability(full_trace, proxy = 'd13c')

plt.show()