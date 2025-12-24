import numpy as np

import tensorly as tl
from tensorly.decomposition import tucker

from external_models_utils import *


class model_tucker():
    def __init__(self, data, K, seed=None,
                 explained_variance_treshold = 0.01,
                 scale_views=True):
        """Tucker decomposition.

        Parameters
        ----------
        data : dict
            Dictionary containing the data for each view.
        K : int
            Number of latent factors to initialize the model with (> 0).
        seed : int, optional
            Random seed for reproducibility. Default is None.
        explained_variance_threshold : float, optional
            Exclude factors whose variance is lower than this threshold. 
        scale_views : bool, optional
            Whether to scale the views before fitting. Default is True.
        """
        
        self.data_keys = list(data.keys())
        self.M = len(self.data_keys)
        # by defaulf center views
        self.data = [(data[self.data_keys[m]] - np.nanmean(data[self.data_keys[m]], axis=0)) for m in range(self.M)]
        if scale_views:
            std_data = [np.nanstd(self.data[m], axis=0) for m in range(self.M)]
            for m in range(self.M):
                std_data[m][std_data[m] == 0] = 1
            self.data = [(self.data[m])/std_data[m] for m in range(self.M)]
        self.data_tensor = np.array(self.data)
        self.data_tensor = np.moveaxis(self.data_tensor, [0,1,2], [1,2,0])
        self.data_tensor = tl.tensor(self.data_tensor)

        self.N = data[self.data_keys[0]].shape[0]
        self.data_list = [data[self.data_keys[m]] for m in range(self.M)]
        self.D = [self.data_list[m].shape[1] for m in range(self.M)]
        self.views_names = self.data_keys
        self.K = K
        self.seed = seed
        self.explained_variance_treshold = explained_variance_treshold

        self.model = None

    def fit(self):
        core, factors = tucker(self.data_tensor, rank=[self.D[0], self.M, self.K])
        self.K_fitted = self.K
        self.factors = factors[2]
        weights = np.tensordot(factors[0], core, axes=([1], [0])).swapaxes(0, 0)
        weights = np.tensordot(factors[1], weights, axes=([1], [1])).swapaxes(0, 1)
        self.weights = [weights[:,m,:] for m in range(self.M)]
        self.model_data = self.data
        self.intercepts = [np.mean(self.data[m], axis=0) for m in range(self.M)]

        
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
        
          
    def get_model(self):
        return self.model
    
    def variance_explained_per_factor_per_view(self):
        
        return explained_variance_factor_view(self.model_data, self.factors, self.weights, self.intercepts)

    
    def variance_explained_per_factor(self):
        
        return explained_variance_factor(self.model_data, self.factors, self.weights, self.intercepts)

    
    def variance_explained_per_view(self):
        
        return explained_variance_view(self.model_data, self.factors, self.weights, self.intercepts)

        
    def variance_explained_total(self):
        
        return explained_variance_total(self.model_data, self.factors, self.weights, self.intercepts)

    
    def get_factors(self):
        return self.factors

    def get_weights(self):
        return self.weights
     
    def get_number_of_active_factors(self):
        return self.K_fitted
