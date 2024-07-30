from stratmc.config import PROJECT_ROOT
from stratmc.data import load_object, load_trace
from stratmc.inference import extend_age_model, interpolate_proxy
from stratmc.plotting import interpolated_proxy_inference

full_trace = load_trace(str(PROJECT_ROOT) + '/examples/example_docs_trace')
example_sample_path = str(PROJECT_ROOT) + '/examples/example_sample_df'
example_sample_path_d18o = str(PROJECT_ROOT) + '/examples/example_sample_df_d18o'
example_ages_path = str(PROJECT_ROOT) + '/examples/example_ages_df'
sample_df = load_object(example_sample_path)
sample_df_d18o = load_object(example_sample_path_d18o)
ages_df = load_object(example_ages_path)

interpolated_df = extend_age_model(full_trace, sample_df, ages_df, ['d18o'], new_proxy_df = sample_df_d18o)
ages_new = full_trace.X_new.X_new.values.ravel()
interpolated_proxy_df = interpolate_proxy(interpolated_df, 'd18o', ages_new)

interpolated_proxy_inference(interpolated_df, interpolated_proxy_df, 'd18o')

plt.show()