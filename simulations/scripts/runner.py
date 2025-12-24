import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import subprocess

file_name = 'param_sim_3.csv'

subprocess.Popen(["python", "run_models_parallel.py", "model_cling_AD", '0.005', 'None', file_name])
subprocess.Popen(["python", "run_models_parallel.py", "model_cling_AD", '0.01', 'None', file_name])
subprocess.Popen(["python", "run_models_parallel.py", "model_cling_AD", '0.02', 'None', file_name])
print('model_cling add delete: completed')

# # subprocess.Popen(["python", "run_models_parallel.py", "model_mofa", '0.005', 'None', file_name])
# # subprocess.Popen(["python", "run_models_parallel.py", "model_mofa", '0.01', 'None', file_name])
# # subprocess.Popen(["python", "run_models_parallel.py", "model_mofa", '0.02', 'None', file_name])
# # print('mofa 1: completed')

# # subprocess.Popen(["python", "run_models_parallel.py", "model_mofa_prune", '0.005', 'None', file_name])
# subprocess.Popen(["python", "run_models_parallel.py", "model_mofa_prune", '0.01', 'None', file_name])
# # subprocess.Popen(["python", "run_models_parallel.py", "model_mofa_prune", '0.02', 'None', file_name])
# # print('mofa 2: completed')

# # subprocess.Popen(["python", "run_models_parallel.py", "model_pca", '0.005', 'None', file_name])
# subprocess.Popen(["python", "run_models_parallel.py", "model_pca", '0.01', 'None', file_name])
# # subprocess.Popen(["python", "run_models_parallel.py", "model_pca", '0.02', 'None', file_name])
# # print('pca: completed')

# # subprocess.Popen(["python", "run_models_parallel.py", "model_tucker", '0.005', 'None', file_name])
# subprocess.Popen(["python", "run_models_parallel.py", "model_tucker", '0.01', 'None', file_name])
# # subprocess.Popen(["python", "run_models_parallel.py", "model_tucker", '0.02', 'None', file_name])
# # print('tucker: completed')

# # subprocess.Popen(["python", "run_models_parallel.py", "model_muvi", '0.005', 'None', file_name])
# subprocess.Popen(["python", "run_models_parallel.py", "model_muvi", '0.01', 'None', file_name])
# # # subprocess.Popen(["python", "run_models_parallel.py", "model_muvi", '0.02', 'None', file_name])
# print('model_muvi: completed')
