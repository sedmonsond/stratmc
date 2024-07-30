from stratmc.config import PROJECT_ROOT
from stratmc.data import load_trace
from stratmc.plotting import age_constraints

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_trace')
section = '1'

age_constraints(full_trace, section)

plt.show()