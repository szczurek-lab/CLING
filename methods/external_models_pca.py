import numpy as np

from sklearn.decomposition import PCA

from external_models_utils import *


class model_pca():
    def __init__(self, data, K,  seed=None,
                 explained_variance_treshold=0.01,
                 scale_views=True):
        """PCA.

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
        self.data = np.concat([(data[self.data_keys[m]] - np.nanmean(data[self.data_keys[m]], axis=0)).T for m in range(self.M)]).T
        if scale_views:
            std_data = np.nanstd(self.data, axis=0)
            std_data[std_data == 0] = 1
            self.data = self.data/std_data
        self.data_list = [data[self.data_keys[m]] for m in range(self.M)]
        self.D = [self.data_list[m].shape[1] for m in range(self.M)]
        self.views_names = self.data_keys
        self.K = K
        self.explained_variance_treshold = explained_variance_treshold
        self.seed = seed

        self.model = PCA(n_components=self.K)

    def fit(self):

        self.model = self.model.fit(self.data)
        if self.explained_variance_treshold is not None:
            self.K_fitted = np.sum(self.model.explained_variance_ratio_ > self.explained_variance_treshold)
        else:
            self.K_fitted = self.K
        self.factors = self.model.transform(self.data)[:,:self.K_fitted]
        self.weights = np.split(self.model.components_[:self.K_fitted,:], np.cumsum(self.D)[:-1], axis=1)
        self.weights = [self.weights[m].T for m in range(self.M)]
        self.model_data = np.split(self.data, np.cumsum(self.D)[:-1], axis=1)
        self.intercepts = [np.mean(self.model_data[m], axis=0) for m in range(self.M)]

        if self.K_fitted == 0:
            self.factors = np.empty((0,0))
            self.weights = [np.empty((0,0)) for m in range(self.M)]
            self.model_data = None
        
          
    def get_model(self):
        return self.model
    
    def variance_explained_per_factor_per_view(self):
        if self.model is not None:
            return explained_variance_factor_view(self.model_data, self.factors, self.weights, self.intercepts)
        else:
            return 0
    
    def variance_explained_per_factor(self):
        if self.model is not None:
            return explained_variance_factor(self.model_data, self.factors, self.weights, self.intercepts)
        else:
            return 0
    
    def variance_explained_per_view(self):
        if self.model is not None:
            return explained_variance_view(self.model_data, self.factors, self.weights, self.intercepts)
        else:
            return 0
        
    def variance_explained_total(self):
        if self.model is not None:
            return explained_variance_total(self.model_data, self.factors, self.weights, self.intercepts)
        else:
            return 0
    
    def get_factors(self):
        return self.factors

    def get_weights(self):
        return self.weights
     
    def get_number_of_active_factors(self):
        return self.K_fitted
