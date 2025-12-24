import numpy as np
import muvi

from external_models_utils import *

class model_muvi():
    def __init__(self, data, K, seed=None,
                 explained_variance_treshold = 0.01,
                 normalize=True, likelihoods=None,
                 prior_masks=None,
                 device="cpu"):
        """Model muVI.

        Parameters
        ----------
        data : dict
            Dictionary containing the data for each view.
        K : int
            Number of latent factors to initialize the model with (> 0).
        seed : int, optional
            Random seed for reproducibility. Default is None.
        normalize : bool, optional
            Whether to normalize the views before fitting (centering and scaling by a global std). Default is True.
            - ML: recommend not changing that
        likelihoods : list of str, optional
            List of likelihood/distribution names for each view. 
            Default is 'normal' for all views.
        prior_masks: None
            Prior on W
            We use fully unsupervised version without prior knowledge on W.
        device: str,
            "cuda" or "cpu"
        """
        self.data_keys = list(data.keys())
        self.M = len(self.data_keys)
        self.data = data
        self.views_names = self.data_keys
        self.K = K
        self.seed = seed

        self._normalize = normalize
        self.explained_variance_treshold = explained_variance_treshold


        self.model = muvi.MuVI(
            observations=self.data,
            n_factors=self.K,
            view_names=self.data_keys,
            prior_masks=prior_masks,
            likelihoods=likelihoods,
            normalize=normalize,
            device=device, 
            )
        
    def fit(self):

        self.model.fit(seed=self.seed)
        print('MODEL FIT FINE 0 ')

        self.factors = self.model.get_factor_scores()
        self.weights = [self.model.get_factor_loadings()[m].T for m in self.data_keys]
        self.K_fitted = self.model.n_factors
        # I assume that they don't fit intercept as they assume the data is normalized
        self.intercepts = None
        if self._normalize:
            data_tmp = self.model.get_observations()
            self.model_data = [(data_tmp[self.data_keys[m]] - np.nanmean(data_tmp[self.data_keys[m]], axis=0))/np.nanstd(data_tmp[self.data_keys[m]]) for m in range(self.M)]
        else:
            self.model_data = [data_tmp[self.data_keys[m]] for m in range(self.M)]

        if self.K_fitted > 0 and self.explained_variance_treshold is not None:
            var_exp_fac = self.variance_explained_per_factor()
            factors_selected = var_exp_fac > self.explained_variance_treshold
            self.factors = self.factors[:,factors_selected]
            self.weights = [self.weights[m][:,factors_selected] for m in range(self.M)]
            self.K_fitted = self.factors.shape[1]

        if self.K_fitted == 0:
            self.factors = np.empty((0,0))
            self.weights = [np.empty((0,0)) for m in range(self.M)]
            self.model_data = None

        print('MODEL FIT FINE')

    def get_model(self):
        return self.model
    
    def variance_explained_per_factor_per_view(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_factor_view(self.model_data, self.factors, self.weights, self.intercepts)
    
    def variance_explained_per_factor(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_factor(self.model_data, self.factors, self.weights, self.intercepts)

    def variance_explained_per_view(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_view(self.model_data, self.factors, self.weights, self.intercepts)
    
    def variance_explained_total(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_total(self.model_data, self.factors, self.weights, self.intercepts)
    
    def get_factors(self):
        return self.factors

    def get_weights(self):
        return self.weights
     
    def get_number_of_active_factors(self):
        return self.K_fitted