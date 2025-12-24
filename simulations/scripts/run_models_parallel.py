max_workers = 16

###### PACKAGES

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import ast

import sys
sys.path.append('../methods')
sys.path.append('../')

###### METHODS
# Comment out the other methods when using MuVI and use the appropriate environment.
from external_models_mofa import model_mofa, model_mofa_prune 
from external_models_pca import model_pca
from external_models_tucker import model_tucker
from internal_models_cling import model_cling_AD, model_cling_ablation1, model_cling_ablation2
# from external_models_muvi import model_muvi
from model_wrappers import *

###### SIMULATION FUNCTION
# from simulate_mofa_v2 import simulate_data
from cling_sparsity_sim import sparsity_data as simulate_data

###### INPUT
model_name = sys.argv[1]
try:
    model_func = globals()[model_name]  # look up function in global namespace
except KeyError:
    raise ValueError(f"Unknown model name: {model_name}")

exp_var_tres = sys.argv[2]
if exp_var_tres == 'None': 
    exp_var_tres = None
else:
    exp_var_tres = float(sys.argv[2])

version = sys.argv[3]
if version == 'None': 
    version = None

sim_param_file_name = sys.argv[4]
sim_param_file_name_save = sim_param_file_name.replace(".csv", "")

###### "ONE" SIMULATION FUNCTION
def run_single_simulation(simulation_params):
    sim_data = simulate_data(**simulation_params)

    # simulate_mofa, simulate_mofa_v2
    # data = sim_data['data']
    # data = {'M'+str(i): data[i][0] for i in range(len(data))}
    # z = sim_data['Z'][0]
    # w = sim_data['W']

    # clingfa_sim
    data = sim_data['data']
    z = sim_data['Z']
    w = sim_data['W']

    print('start')
    results = run_models(data, z, w, model_func, params={'seed': simulation_params['seed'],
                                                         'explained_variance_treshold': exp_var_tres},
                                                          version = version)
    print('Fitted')

    combined = {**simulation_params, **results}
    return combined

###### READ DATA
df = pd.read_csv(sim_param_file_name, index_col=0)
df['D'] = df['D'].apply(ast.literal_eval)
param_dicts = [df.iloc[i].to_dict() for i in range(df.shape[0])]

if exp_var_tres is None:
    if version is None:
        output_file = sim_param_file_name_save+"/"+model_name+".csv"
    else:
        output_file = sim_param_file_name_save+"/"+model_name+'_'+version+".csv"
else:
    if version is None:
        output_file = sim_param_file_name_save+"/"+model_name+'_'+str(exp_var_tres)+".csv"
    else:
        output_file = sim_param_file_name_save+"/"+model_name+'_'+str(exp_var_tres)+'_'+version+".csv"

###### RUN SIMULATIONS
# Run in parallel and save partial results
# with ProcessPoolExecutor(max_workers=max_workers) as executor:
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    futures = [executor.submit(run_single_simulation, p) for p in param_dicts]

    # Process completed tasks one by one
    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()
        # Append to CSV
        pd.DataFrame([result]).to_csv(output_file, mode='a', header=not pd.io.common.file_exists(output_file), index=False)