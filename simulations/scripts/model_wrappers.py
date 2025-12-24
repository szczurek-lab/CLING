import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import jaccard_score
import sys

sys.path.append('../methods')
sys.path.append('../')

# Comment out the other methods when using MuVI and use the appropriate environment.
from external_models_mofa import model_mofa, model_mofa_prune
from external_models_pca import model_pca
from external_models_tucker import model_tucker
from internal_models_cling import model_cling_AD, model_cling_ablation1, model_cling_ablation2
# from external_models_muvi import model_muvi


def compute_correlation_factors(z, z_est):
    if z.shape[1] == 0:
        return 0
    if z is None:
        return 0
    if z_est.shape[1] != 0:
        cor_all = []
        for k in range(z.shape[1]):
            z_k = z[:,k]
            cor_k = np.max([np.abs(spearmanr(z_est[:,k2], z_k)[0]) for k2 in range(z_est.shape[1])])
            cor_all.append(cor_k)
        return np.mean(cor_all)
    else:
        return 0
    
def compute_corr_jaccard_weights(z, z_est, w, w_est, tres_w=0.9):
    if z.shape[1] == 0:
        return 0, 0
    if z is None:
        return 0, 0
    if z_est is None:
        return 0, 0
    if z_est.shape[1] != 0:
        jaccard_index_per_factor = []
        corr_per_factor = []
        for k in range(z.shape[1]):
            z_k = z[:,k]
            k_z_est =  np.argmax([np.abs(spearmanr(z_est[:,k2], z_k)[0]) for k2 in range(z_est.shape[1])])
            for m in range(len(w)):
                w_tmp = np.abs(w[m][:,k])
                w_est_tmp = np.abs(w_est[m][:,k_z_est])
                big_loadings = (w_tmp > np.quantile(w_tmp, tres_w))
                big_loadings_est = w_est_tmp > np.quantile(w_est_tmp, tres_w)
                jaccard_score_tmp = jaccard_score(big_loadings, big_loadings_est)
                jaccard_index_per_factor.append(jaccard_score_tmp)

                w_tmp = w[m][:,k]
                w_est_tmp = w_est[m][:,k_z_est]
                corr_tmp = np.abs(spearmanr(w_tmp, w_est_tmp)[0])
                corr_per_factor.append(corr_tmp)
        return np.mean(jaccard_index_per_factor), np.mean(corr_per_factor)
    else:
        return 0, 0
    

def run_models(data, z, w, model, params={}, version=None):
    K_trunc = int(30)
    N = data['M'+str(0)].shape[0]
    M = len(data)
    D = [data['M'+str(m)].shape[1] for m in range(M)]

    if version is None:
        mod = model(data, K_trunc, **params)
    else:
        mod = model(data, K_trunc, version=version, **params)
    
    mod.fit()

    K_est = mod.get_number_of_active_factors()
    mean_var_expl = np.mean(mod.variance_explained_per_view())
    var_expl = mod.variance_explained_total()
    cor_factors = compute_correlation_factors(z, mod.get_factors())   
    jac_weights, cor_weights = compute_corr_jaccard_weights(z, mod.get_factors(), w, mod.get_weights())

    
    if params['explained_variance_treshold'] is not None:
        exp_var_name = '_' + str(params['explained_variance_treshold'])
        tres = mod.explained_variance_treshold
    else:
        exp_var_name = ''
        tres = None

    if version is None:
        model_name = model.__name__+exp_var_name
    else:
        model_name = model.__name__+exp_var_name+"_"+version

    results = {
        'K_est': K_est,
        'Var_exp_views': mean_var_expl,
        'Var_exp_total': var_expl,
        'Cor_factors': cor_factors,
        'model': model_name,
        'tres': tres,
        'Jaccard_index': jac_weights,
        'Cor_weights': cor_weights
    }

    return results