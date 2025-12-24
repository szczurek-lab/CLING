import numpy as np
import os
import sys
from unittest.mock import patch

from external_models_utils import *
sys.path.append('../')

# load CLING model and ablations
from cling import ClingFA
from cling_ablation1 import ClingFA_ablation1
from cling_ablation2 import ClingFA_ablation2

class model_cling_AD():
    def __init__(self, data, K, seed=None, explained_variance_treshold=None,
                 max_iter=2000, prune_every=25, center_data=True):
        """CLING add-delete

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
        max_iter : int
            Maximum number of iterations. Default 2000.
        prune_every: int
            How often should pruning be done (every how many iterations). Default 25.
        center_data: bool
            Whether to center the views before fitting. Default is True.
        """
        data_keys = list(data.keys())
        self.views_names = data_keys
        self.M = len(data_keys)
        self.N = data['M0'].shape[0]

        # model input
        self._center_data = center_data
        if self._center_data:
            # centering
            self.data = {'M'+str(m): data[data_keys[m]] - np.nanmean(data[data_keys[m]], axis=0) for m in range(self.M)}
        else:
            self.data = data

        # simple data storage - easy to use
        self.data_list = [data[data_keys[m]] for m in range(self.M)]
        
        self.K = K
        self.seed = seed
        self.D = [data[data_keys[m]].shape[1] for m in range(self.M)]

        self._center_data = center_data

        self.explained_variance_treshold = explained_variance_treshold
        if self.explained_variance_treshold is None:
            self.explained_variance_treshold = 0.01

        # set seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Cling init
        self.model = ClingFA.from_numpy_views(
                                            data,
                                            K=None,
                                            initK_mode="Mlogmax",
                                            initK_c=5.0,
                                            mgp_a1=2, mgp_a2=2.1, mgp_b1=1, mgp_b2=1,
                                            ard_a_alpha=0.5, ard_b_alpha=1.0,
                                            a_phi=0.5, b_phi=1.0,
                                            center=[True]*len(data),
                                            init_mode="pca",
                                            prune_mode='per_view',
                                            prune_threshold=self.explained_variance_treshold,
                                            prune_min_views=1
                                        )
        
        self._max_iter = max_iter
        self._prune_every = prune_every

    def fit(self, add_k=3):

        _, _, _ = self.model.fit(
            max_iter=self._max_iter,
            tol=1e-10,
            prune_warmup=100,
            prune_every=self._prune_every,
            add_every=self._prune_every,
            add_patience=2,
            add_k=add_k,
            target_view_R2=0.99,
            min_views_below=1,
            max_K=200,
            verbose=True,
            two_value_lock=True,
            two_value_history_len=8,
            two_value_min_len=6,
            lock_extra_iters=50,
            oscillation_guard=True,
            oscillation_flips=10,
            oscillation_window = 12
        )

        self.K_fitted = self.model.K
        self.factors = self.model.get_factors()
        self.weights = self.model.get_weights()
        if self._center_data:
            # centering
            self.model_data = [self.data_list[m] - np.mean(self.data_list[m], axis=0) for m in range(self.M)]
        else:
            self.model_data = [self.data_list[m] for m in range(self.M)]     

        if self.K_fitted == 0:
            self.factors = np.empty((0,0))
            self.weights = [np.empty((0,0)) for m in range(self.M)]
            self.model_data = None

          
    def get_model(self):
        return self.model
    
    def variance_explained_per_factor_per_view(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_factor_view(self.model_data, self.factors, self.weights)
    
    def variance_explained_per_factor(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_factor(self.model_data, self.factors, self.weights)
    
    def variance_explained_per_view(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_view(self.model_data, self.factors, self.weights)
    
    def variance_explained_total(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_total(self.model_data, self.factors, self.weights)
    
    def get_factors(self):
        return self.factors

    def get_weights(self):
        return self.weights
     
    def get_number_of_active_factors(self):
        return self.K_fitted
    

class model_cling_ablation1():
    def __init__(self, data, K, seed=None, explained_variance_treshold=None,
                 max_iter=2000, prune_every=25, center_data=True,
                 version='a'):
        """CLING ablation 1: MGP

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
        max_iter : int
            Maximum number of iterations. Default 2000.
        prune_every: int
            How often should pruning be done (every how many iterations). Default 25.
        center_data: bool
            Whether to center the views before fitting. Default is True.
        """
        data_keys = list(data.keys())
        self.views_names = data_keys
        self.M = len(data_keys)
        self.N = data['M0'].shape[0]

        # model input
        self._center_data = center_data
        if self._center_data:
            # centering
            self.data = {'M'+str(m): data[data_keys[m]] - np.nanmean(data[data_keys[m]], axis=0) for m in range(self.M)}
        else:
            self.data = data

        # simple data storage - easy to use
        self.data_list = [data[data_keys[m]] for m in range(self.M)]
        
        self.K = K
        self.seed = seed
        self.D = [data[data_keys[m]].shape[1] for m in range(self.M)]

        self._center_data = center_data

        self.explained_variance_treshold = explained_variance_treshold
        if self.explained_variance_treshold is None:
            self.explained_variance_treshold = 0.01

        # set seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Cling init
        self.model = ClingFA_ablation1.from_numpy_views(
                                            data, 
                                            K=self.K,
                                            mgp_a1=2, mgp_a2=2.1, mgp_b1=1, mgp_b2=1,
                                            ard_a_alpha=1.5, ard_b_alpha=1.5,
                                            center=[True]*len(data),
                                            prune_mode='per_view',
                                            prune_threshold=self.explained_variance_treshold,
                                            prune_min_views=1
                                        )
        
        self._max_iter = max_iter
        self._prune_every = prune_every

    def fit(self, add_k=3):

        _, _, _ = self.model.fit(
            max_iter=self._max_iter,
            tol=1e-10,
            prune_warmup=100,
            prune_every=self._prune_every,
            add_every=self._prune_every,
            add_patience=2,
            add_k=add_k,
            target_view_R2=0.99,
            min_views_below=1,
            max_K=200,
            verbose=True,
            two_value_lock=True,
            two_value_history_len=8,
            two_value_min_len=6,
            lock_extra_iters=50,
            oscillation_guard=True,
            oscillation_flips=10,
            oscillation_window=12
        )

        self.K_fitted = self.model.K
        self.factors = self.model.get_factors()
        self.weights = self.model.get_weights()
        if self._center_data:
            # centering
            self.model_data = [self.data_list[m] - np.mean(self.data_list[m], axis=0) for m in range(self.M)]
        else:
            self.model_data = [self.data_list[m] for m in range(self.M)]     

        if self.K_fitted == 0:
            self.factors = np.empty((0,0))
            self.weights = [np.empty((0,0)) for m in range(self.M)]
            self.model_data = None

          
    def get_model(self):
        return self.model
    
    def variance_explained_per_factor_per_view(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_factor_view(self.model_data, self.factors, self.weights)
    
    def variance_explained_per_factor(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_factor(self.model_data, self.factors, self.weights)
    
    def variance_explained_per_view(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_view(self.model_data, self.factors, self.weights)
    
    def variance_explained_total(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_total(self.model_data, self.factors, self.weights)
    
    def get_factors(self):
        return self.factors

    def get_weights(self):
        return self.weights
     
    def get_number_of_active_factors(self):
        return self.K_fitted
    


class model_cling_ablation2():
    def __init__(self, data, K, seed=None, explained_variance_treshold=None,
                 max_iter=2000, prune_every=25, center_data=True):
        """CLING ablation 2: ARD

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
        max_iter : int
            Maximum number of iterations. Default 2000.
        prune_every: int
            How often should pruning be done (every how many iterations). Default 25.
        center_data: bool
            Whether to center the views before fitting. Default is True.
        """
        data_keys = list(data.keys())
        self.views_names = data_keys
        self.M = len(data_keys)
        self.N = data['M0'].shape[0]

        # model input
        self._center_data = center_data
        if self._center_data:
            # centering
            self.data = {'M'+str(m): data[data_keys[m]] - np.nanmean(data[data_keys[m]], axis=0) for m in range(self.M)}
        else:
            self.data = data

        # simple data storage - easy to use
        self.data_list = [data[data_keys[m]] for m in range(self.M)]
        
        self.K = K
        self.seed = seed
        self.D = [data[data_keys[m]].shape[1] for m in range(self.M)]

        self._center_data = center_data

        self.explained_variance_treshold = explained_variance_treshold
        if self.explained_variance_treshold is None:
            self.explained_variance_treshold = 0.01

        # set seed
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Cling init
        self.model = ClingFA_ablation2.from_numpy_views(
                                            data,
                                            K=self.K,
                                            mgp_a1=2, mgp_a2=2.1, mgp_b1=1, mgp_b2=1,
                                            ard_a_alpha=1.5, ard_b_alpha=1.5,
                                            center=[True]*len(data),
                                            prune_mode='per_view',
                                            prune_threshold=self.explained_variance_treshold,
                                            prune_min_views=1
                                        )
        
        self._max_iter = max_iter
        self._prune_every = prune_every

    def fit(self, add_k=3):

        _, _, _ = self.model.fit(
            max_iter=self._max_iter,
            tol=1e-10,
            prune_warmup=100,
            prune_every=self._prune_every,
            add_every=self._prune_every,
            add_patience=2,
            add_k=add_k,
            target_view_R2=0.99,
            min_views_below=1,
            max_K=200,
            verbose=True,
            two_value_lock=True,
            two_value_history_len=8,
            two_value_min_len=6,
            lock_extra_iters=50,
            oscillation_guard=True,
            oscillation_flips=10,
            oscillation_window=12
        )

        self.K_fitted = self.model.K
        self.factors = self.model.get_factors()
        self.weights = self.model.get_weights()
        if self._center_data:
            # centering
            self.model_data = [self.data_list[m] - np.mean(self.data_list[m], axis=0) for m in range(self.M)]
        else:
            self.model_data = [self.data_list[m] for m in range(self.M)]     

        if self.K_fitted == 0:
            self.factors = np.empty((0,0))
            self.weights = [np.empty((0,0)) for m in range(self.M)]
            self.model_data = None

          
    def get_model(self):
        return self.model
    
    def variance_explained_per_factor_per_view(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_factor_view(self.model_data, self.factors, self.weights)
    
    def variance_explained_per_factor(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_factor(self.model_data, self.factors, self.weights)
    
    def variance_explained_per_view(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_view(self.model_data, self.factors, self.weights)
    
    def variance_explained_total(self):
        if self.K_fitted == 0:
            return 0
        else:
            return explained_variance_total(self.model_data, self.factors, self.weights)
    
    def get_factors(self):
        return self.factors

    def get_weights(self):
        return self.weights
     
    def get_number_of_active_factors(self):
        return self.K_fitted
  