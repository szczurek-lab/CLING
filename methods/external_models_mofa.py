import numpy as np
import os

import sys
from unittest.mock import patch

import muon as mu
import anndata as ad

import mofax as mfx
from mofapy2.run.entry_point import entry_point

from external_models_utils import *


class model_mofa_general():
    def __init__(self, data, K, seed=None,
                 explained_variance_treshold=0.01, prune=True,
                 scale_views=False, likelihoods=None):
        """MOFA from the original package mofapy2.
        For simplicity, please use model_mofa or model_mofa_prune below.

        Parameters
        ----------
        data : dict
            Dictionary containing the data for each view.
        K : int
            Number of latent factors to initialize the model with (> 0).
        seed : int, optional
            Random seed for reproducibility. Default is None.
        explained_variance_threshold : float, optional
            Exclude factors whose variance is lower than this threshold. Default is None (include all factors) (0 < and < 1).
        scale_views : bool, optional
            Whether to scale the views before fitting. Default is False.
        likelihoods : list of str, optional
            List of likelihood/distribution names for each view. 
            Default is 'normal' for all views.
        """
        data_keys = list(data.keys())
        self.views_names = data_keys
        self.M = len(data_keys)

        # model input
        # centering as MOFA would do this anyway
        self.data = [[data[data_keys[m]] - np.nanmean(data[data_keys[m]], axis=0)] for m in range(self.M)]

        # simple data storage - easy to use
        self.data_list = [data[data_keys[m]] for m in range(self.M)]
        
        self.K = K
        self.explained_variance_treshold = explained_variance_treshold
        self.seed = seed
        self._prune = prune

        # mofapy2 init
        self.model = entry_point()
        self.model.set_data_options(scale_views=scale_views)
        self.model.set_data_matrix(self.data, 
                                   likelihoods=likelihoods, views_names=self.views_names,
                                   samples_names=None, features_names=None)

    def fit(self,
            ard_weights=True, spikeslab_weights=True, 
            ard_factors=False, spikeslab_factors=False):

        self.model.set_model_options(
            factors=self.K,
            ard_factors=ard_factors, ard_weights=ard_weights,
            spikeslab_factors=spikeslab_factors, spikeslab_weights=spikeslab_weights
            )
        
        # pruning during/after training
        if self._prune:
            self.model.set_train_options(
                dropR2=self.explained_variance_treshold,
                quiet=True, seed=self.seed
            )
        else:
            self.model.set_train_options(
                quiet=True, seed=self.seed
            )
        self.model.build()

        try: # model sometimes crushes if all factors are inactive
            try: # and sometimes it exits when all factors are inactive
                self.model.run()
                self.factors = self.model.model.getExpectations()['Z']['E']
                self.weights = [self.model.model.getExpectations()['W'][i]['E'] for i in range(self.M)]
                self.K_fitted = self.model.model.dim['K']
                self.model_data = self.model.data
            except SystemExit:
                self.model = None
                self.factors = np.empty((0,0))
                self.weights = [np.empty((0,0)) for m in range(self.M)]
                self.K_fitted = 0
                self.model_data = None
        except Exception:
            self.model = None
            self.factors = np.empty((0,0))
            self.weights = [np.empty((0,0)) for m in range(self.M)]
            self.K_fitted = 0
            self.model_data = None

        # pruning after training
        if (self.K_fitted > 0 and not self._prune) and self.explained_variance_treshold is not None:
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
        if self.K_fitted > 0:
            return explained_variance_factor_view(self.model_data, self.factors, self.weights)
        else:
            return 0
    
    def variance_explained_per_factor(self):
        if self.K_fitted > 0:
            return explained_variance_factor(self.model_data, self.factors, self.weights)
        else:
            return 0
    
    def variance_explained_per_view(self):
        if self.K_fitted > 0:
            return explained_variance_view(self.model_data, self.factors, self.weights)
        else:
            return 0
    
    def variance_explained_total(self):
        if self.K_fitted > 0:
            return explained_variance_total(self.model_data, self.factors, self.weights)
        else:
            return 0
    
    def get_factors(self):
        return self.factors

    def get_weights(self):
        return self.weights
     
    def get_number_of_active_factors(self):
        return self.K_fitted

class model_mofa_prune(model_mofa_general):
    """MOFA.
        This MOFA does pruning during training
    """
    def __init__(self, data, K, seed=None,
                 explained_variance_treshold=0.01):
        super().__init__(data=data, K=K, seed=seed, explained_variance_treshold=explained_variance_treshold,
                         prune=True)
        
class model_mofa(model_mofa_general):
    """MOFA.
        This MOFA does pruning after training
    """
    def __init__(self, data, K, seed=None,
                 explained_variance_treshold=0.01):
        super().__init__(data=data, K=K, seed=seed, explained_variance_treshold=explained_variance_treshold, 
                         prune=False)