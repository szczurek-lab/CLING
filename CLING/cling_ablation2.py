######################## CLING-Ablation2 ########################
# Variant without phi: alpha shared per (m, k); single global MGP over k (shared across views).

from __future__ import annotations
import numpy as np
from scipy.special import digamma, gammaln
from typing import Dict, List, Optional, Tuple

try:
    import plotly.graph_objects as go
except Exception:
    go = None

__all__ = ["ClingFA_ablation2", "compute_init_K"]

EPS = 1e-10
def log_eps(x, eps=EPS): return np.log(np.maximum(x, eps))


# ------------------------ initial K helper ------------------------
def compute_init_K(D: List[int], M: int, mode: str = "paper", c: float = 5.0,
                   floor: int = 3, cap: int = 2000) -> int:
    p = int(np.sum(D))
    dmax = int(np.max(D))
    if mode == "Mlogmax":
        K0 = int(np.ceil(M * max(1.0, np.log(max(dmax, 2)))))
    else:
        K0 = int(np.ceil(c * max(1.0, np.log(max(p, 2)))))
    return int(np.clip(K0, floor, cap))


# ------------------------ base ------------------------
class nodeFA_general:
    def __init__(self, N, K, D, M):
        self.N = N
        self.K = K
        self.D = D
        self.M = M


# ------------------------ Z ------------------------
class nodeFA_z(nodeFA_general):
    def __init__(self, vi_mu, vi_var, general_params):
        super().__init__(**general_params)
        self.prior_mu = 0.0
        self.prior_var = 1.0
        self.vi_mu = vi_mu
        self.vi_var = np.maximum(vi_var, 1e-8)
        self.update_params()
        self.elbo = 0.0

    def MB(self, y_list, w_list, tau_list):
        self.y_node = y_list
        self.w_node = w_list
        self.tau_node = tau_list

    def update_k(self, k):
        vi_mu_new = np.zeros(self.N)
        vi_var_new = np.zeros(self.N)

        for m in range(self.M):
            Y = self.y_node[m].Y
            Msk = self.y_node[m].M
            E_tau = self.tau_node[m].E_tau
            Wm = self.w_node[m].E_w

            vi_var_new += (Msk * (E_tau * self.w_node[m].E_w_squared[:, k])[None, :]).sum(axis=1)

            WZ = (Wm @ self.E_z.T).T
            resid = Msk * (Y - WZ)
            partial = resid + Msk * np.outer(self.E_z[:, k], Wm[:, k])

            vi_mu_new += (Msk * (E_tau * Wm[:, k])[None, :] * partial).sum(axis=1)

        vi_var_new = 1.0 / np.maximum(vi_var_new + 1.0, 1e-8)
        self.vi_var[:, k] = vi_var_new
        self.vi_mu[:, k] = vi_mu_new * vi_var_new
        self.update_params()

    def update_params(self):
        self.vi_var = np.maximum(self.vi_var, 1e-8)
        self.E_z = self.vi_mu
        self.E_z_squared = self.vi_var + self.vi_mu**2

    def ELBO(self):
        self.elbo = self.N*self.K/2 - np.sum(self.E_z_squared)/2 + np.sum(log_eps(self.vi_var, EPS))/2


# ------------------------ W ------------------------
class nodeFA_w_m(nodeFA_general):
    """
    q(W^{(m)}) = N(vi_mu, diag(vi_var))
    Prior: W_{d,k}^{(m)} ~ N(0, 1 / (alpha_{k}^{(m)} * gamma_k))
    """
    def __init__(self, m, w_mu, w_var, general_params):
        super().__init__(**general_params)
        self.m = m
        self.vi_mu = w_mu.copy()
        self.vi_var = np.maximum(w_var.copy(), 1e-8)
        self.E_w = self.vi_mu
        self.E_w_squared = self.vi_var + self.vi_mu**2
        self.E_w_z = np.zeros((self.N, self.D[self.m]))
        self.E_w_z_squared = np.zeros((self.N, self.D[self.m]))
        self.elbo = 0.0

    def MB(self, alpha_m_node, z_node, y_m_node, tau_m_node, delta_global_node):
        self.alpha_m_node = alpha_m_node
        self.z_node = z_node
        self.y_m_node = y_m_node
        self.tau_m_node = tau_m_node
        self.delta_global_node = delta_global_node
        self.update_params()
        self.update_params_z()

    def update_k(self, k):
        E_tau = self.tau_m_node.E_tau                # (D_m,)
        E_alpha_k = self.alpha_m_node.E_alpha[k]     # scalar (per view)
        E_gamma_k = self.delta_global_node.E_gamma[k]# scalar (global)
        Z = self.z_node.E_z
        Zk = Z[:, k]
        Y = self.y_m_node.Y
        Msk = self.y_m_node.M

        WZ = (self.E_w @ Z.T).T
        resid = Msk * (Y - WZ)
        partial = resid + Msk * np.outer(Zk, self.E_w[:, k])

        nominator = Zk @ partial                        # (D_m,)
        sum_z2_masked = (Msk * self.z_node.E_z_squared[:, k][:, None]).sum(axis=0)  # (D_m,)

        prior_prec = E_alpha_k * E_gamma_k              # scalar
        denom = np.maximum(sum_z2_masked + prior_prec / E_tau, 1e-8)

        self.vi_mu[:, k] = nominator / denom
        self.vi_var[:, k] = np.maximum(1.0 / (E_tau * denom), 1e-8)

        self.update_params()
        self.update_params_z()

    def update_params(self):
        self.vi_var = np.maximum(self.vi_var, 1e-8)
        self.E_w = self.vi_mu
        self.E_w_squared = self.vi_var + self.vi_mu**2

    def update_params_z(self):
        self.E_w_z = (self.E_w @ self.z_node.E_z.T).T
        term_tmp = self.E_w @ self.z_node.E_z.T
        first_term = term_tmp**2
        second_term = self.E_w_squared @ self.z_node.E_z_squared.T
        third_term = (self.E_w**2) @ (self.z_node.E_z.T**2)
        self.E_w_z_squared = (first_term + second_term - third_term).T

    def ELBO(self):
        E_log_alpha = self.alpha_m_node.E_log_alpha      # (K,)
        E_log_gamma = self.delta_global_node.E_log_gamma # (K,)
        E_alpha = self.alpha_m_node.E_alpha              # (K,)
        E_gamma = self.delta_global_node.E_gamma         # (K,)

        Dm = self.D[self.m]
        sum_Ew2_per_k = self.E_w_squared.sum(axis=0)     # (K,)

        logp = 0.5 * Dm * np.sum(E_log_alpha + E_log_gamma) \
               - 0.5 * np.sum((E_alpha * E_gamma) * sum_Ew2_per_k)

        entropy = 0.5 * self.D[self.m]*self.K + 0.5 * np.sum(log_eps(self.vi_var, EPS))
        self.elbo = logp + entropy


# ------------------------ Global MGP (delta, gamma) ------------------------
class nodeMGP_delta_global(nodeFA_general):
    """
    Single global MGP over k (shared across all views).
    """
    def __init__(self, a1, b1, a2, b2, K, totalD, general_params):
        super().__init__(**general_params)
        self.a1, self.b1, self.a2, self.b2 = float(a1), float(b1), float(a2), float(b2)
        self.K = K
        self.totalD = int(totalD)

        self.vi_a = np.full(self.K, self.a2, dtype=float); self.vi_a[0] = self.a1
        self.vi_b = np.full(self.K, self.b2, dtype=float); self.vi_b[0] = self.b1

        self.E_delta = np.ones(self.K); self.E_log_delta = np.zeros(self.K)
        self.E_gamma = np.ones(self.K); self.E_log_gamma = np.zeros(self.K)
        self.E_inv_delta = np.ones(self.K); self.E_inv_gamma = np.ones(self.K)
        self.elbo = 0.0
        self.update_params()

    def MB(self, w_nodes: List[nodeFA_w_m], alpha_nodes: List['nodeFA_alpha_m']):
        self.w_nodes = w_nodes
        self.alpha_nodes = alpha_nodes

    def _ak_bk(self, k): return (self.a1, self.b1) if k == 0 else (self.a2, self.b2)

    def _aggregate_S(self) -> np.ndarray:
        """
        S_j = sum_m E[α^{(m)}_j] * sum_d E[W^{(m)2}_{d,j}], shape (K,)
        """
        K = self.K
        S = np.zeros(K, dtype=float)
        for m in range(len(self.w_nodes)):
            Ew2 = self.w_nodes[m].E_w_squared.sum(axis=0)   # (K,)
            Ealpha = self.alpha_nodes[m].E_alpha            # (K,)
            S += Ealpha * Ew2
        return S

    def update_all(self):
        S = self._aggregate_S()
        for k in range(self.K):
            a_k, b_k = self._ak_bk(k)
            self.vi_a[k] = a_k + ((self.K - k) * self.totalD) / 2.0
            rate = b_k
            # sum over j >= k of (E_gamma_j / E_delta_k) * S_j
            egamma_over_edelta = self.E_gamma / max(self.E_delta[k], 1e-12)
            rate += 0.5 * np.sum(egamma_over_edelta[k:] * S[k:])
            self.vi_b[k] = rate
        self.update_params()

    def update_params(self):
        self.vi_b = np.maximum(self.vi_b, 1e-8)
        min_shape = 1.0001
        vi_a_safe = np.maximum(self.vi_a, min_shape)
        self.E_delta = self.vi_a / self.vi_b
        self.E_log_delta = digamma(self.vi_a) - log_eps(self.vi_b, EPS)
        self.E_gamma = np.cumprod(self.E_delta)
        self.E_log_gamma = np.cumsum(self.E_log_delta)
        self.E_inv_delta = self.vi_b / (vi_a_safe - 1.0)
        self.E_inv_gamma = np.cumprod(self.E_inv_delta)

    def ELBO(self):
        prior = 0.0; entropy = 0.0
        for k in range(self.K):
            a_k, b_k = self._ak_bk(k)
            prior += (a_k - 1) * self.E_log_delta[k] - b_k * self.E_delta[k] \
                     + a_k * np.log(b_k + EPS) - gammaln(a_k)
            entropy += ( self.vi_a[k] * log_eps(self.vi_b[k], EPS)
                       + (self.vi_a[k] - 1) * self.E_log_delta[k]
                       - self.vi_b[k] * self.E_delta[k] - gammaln(self.vi_a[k]) )
        self.elbo = prior - entropy


# ------------------------ alpha ------------------------
class nodeFA_alpha_m(nodeFA_general):
    """
    q(α^{(m)}_{k}) = Ga(vi_a[k], vi_b[k]),
    Prior: α^{(m)}_{k} ~ Ga(a_α, b_α)
    Shared across all features d in view m (no φ).
    """
    def __init__(self, a_alpha, b_alpha, m, general_params):
        super().__init__(**general_params)
        self.m = m; self.a_alpha = float(a_alpha); self.b_alpha = float(b_alpha)
        self.Dm = self.D[m]
        self.vi_a = np.full(self.K, self.a_alpha + 0.5*self.Dm, dtype=float)
        self.vi_b = np.full(self.K, max(self.b_alpha, 1e-8), dtype=float)
        self.update_params(); self.elbo = 0.0

    def MB(self, w_m_node, delta_global_node):
        self.w_m_node = w_m_node; self.delta_global_node = delta_global_node

    def update(self):
        E_gamma = self.delta_global_node.E_gamma           # (K,) global
        sum_EW2_per_k = self.w_m_node.E_w_squared.sum(axis=0)  # (K,)
        self.vi_a = np.full(self.K, self.a_alpha + 0.5*self.Dm, dtype=float)
        self.vi_b = self.b_alpha + 0.5 * (E_gamma * sum_EW2_per_k)
        self.update_params()

    def update_k(self, k):
        E_gamma_k = self.delta_global_node.E_gamma[k]
        sum_EW2_k = self.w_m_node.E_w_squared[:, k].sum()
        self.vi_a[k] = self.a_alpha + 0.5*self.Dm
        self.vi_b[k] = self.b_alpha + 0.5 * E_gamma_k * sum_EW2_k
        self.E_alpha[k] = self.vi_a[k] / np.maximum(self.vi_b[k], 1e-8)
        self.E_log_alpha[k] = digamma(self.vi_a[k]) - log_eps(self.vi_b[k], EPS)

    def update_params(self):
        self.vi_b = np.maximum(self.vi_b, 1e-8)
        min_shape = 1.0001
        vi_a_safe2 = np.maximum(self.vi_a, min_shape)
        self.E_alpha = self.vi_a / self.vi_b
        self.E_log_alpha = digamma(self.vi_a) - log_eps(self.vi_b, EPS)
        self.E_inv_alpha = self.vi_b / (vi_a_safe2 - 1.0)

    def ELBO(self):
        a0 = self.a_alpha; b0 = self.b_alpha
        E_log_alpha = self.E_log_alpha; E_alpha = self.E_alpha
        logp = self.K * (a0 * np.log(b0 + EPS) - gammaln(a0)) \
               + (a0 - 1.0) * np.sum(E_log_alpha) - b0 * np.sum(E_alpha)
        logq = np.sum(self.vi_a * log_eps(self.vi_b, EPS) - gammaln(self.vi_a)
                      + (self.vi_a - 1.0) * E_log_alpha - self.vi_b * E_alpha)
        self.elbo = logp - logq


# ------------------------ tau ------------------------
class nodeFA_tau_m(nodeFA_general):
    def __init__(self, a0, b0, m, general_params):
        super().__init__(**general_params)
        self.a0 = a0; self.b0 = b0; self.m = m
        self.vi_a = a0 + self.N * np.ones(self.D[self.m]) / 2.0
        self.vi_b = self.vi_a + 0.0
        self.E_resid_squared_half = 0.0
        self.elbo = 0.0
        self._initialized_constants = False

    def MB(self, y_m_node, w_m_node, z_node):
        self.w_m_node = w_m_node; self.y_m_node = y_m_node; self.z_node = z_node
        obs_count = self.y_m_node.obs_count
        self.vi_a = self.a0 + obs_count / 2.0
        self.vi_b = np.maximum(self.vi_b, 1e-8) if hasattr(self, "vi_b") else self.vi_a + 0.0
        self.update_all_params()
        self._initialized_constants = True

    def update(self):
        self.update_params_w_z()
        self.vi_b = self.b0 + self.E_resid_squared_half
        self.update_params()

    def update_all_params(self):
        self.log_gamma_a0 = gammaln(self.a0)
        self.log_gamma_vi_a = gammaln(self.vi_a)
        self.digamma_vi_a = digamma(self.vi_a)
        self.update_params()
        self.kl_const = -self.D[self.m]*(self.log_gamma_a0) + self.D[self.m]*(self.a0*log_eps(self.b0, EPS))
        self.entropy_cons = np.sum(self.vi_a) + np.sum(self.log_gamma_vi_a) + np.sum((1 - self.vi_a)*self.digamma_vi_a)

    def update_params(self):
        self.vi_b = np.maximum(self.vi_b, 1e-8)
        self.E_tau = self.vi_a / self.vi_b
        self.E_log_tau = -log_eps(self.vi_b, EPS) + self.digamma_vi_a

    def update_params_w_z(self):
        Y = self.y_m_node.Y
        Msk = self.y_m_node.M
        E_wz = self.w_m_node.E_w_z
        E_wz_sq = self.w_m_node.E_w_z_squared
        first = 0.5 * np.sum(Msk * (Y**2), axis=0)
        second = - np.sum(Msk * (Y * E_wz), axis=0)
        third = 0.5 * np.sum(Msk * E_wz_sq, axis=0)
        self.E_resid_squared_half = first + second + third

    def ELBO(self):
        kl = self.kl_const + (self.a0 - 1) * np.sum(self.E_log_tau) - self.b0 * np.sum(self.E_tau)
        entropy = self.entropy_cons - np.sum(log_eps(self.vi_b, EPS))
        self.elbo = kl + entropy


# ------------------------ y ------------------------
class nodeFA_y_m(nodeFA_general):
    def __init__(self, data_n, m, general_params):
        super().__init__(**general_params)
        self.m = m
        self.data = data_n
        self.M = np.isfinite(self.data).astype(float)
        self.Y = np.nan_to_num(self.data, 0.0)
        self.obs_count = self.M.sum(axis=0)
        self.elbo = 0.0
        self.data_mean: Optional[np.ndarray] = None

    def MB(self, w_m_node, tau_m_node):
        self.w_m_node = w_m_node; self.tau_m_node = tau_m_node

    def ELBO(self):
        log2pi = log_eps(2*np.pi, EPS)
        const_term = -0.5 * np.sum(self.obs_count) * log2pi
        tau_log_term = 0.5 * np.sum(self.obs_count * self.tau_m_node.E_log_tau)
        quad_term = np.sum(self.tau_m_node.E_tau * self.tau_m_node.E_resid_squared_half)
        self.elbo = const_term + tau_log_term - quad_term


# ------------------------ helpers ------------------------
def starting_params_z(starting_params, N, K):
    z_mean = 1 * starting_params['z_mu'] if 'z_mu' in starting_params else np.random.normal(0, 0.1, size=(N, K))
    z_var  = starting_params.get('z_var', np.ones((N, K)))
    return z_mean, z_var

def starting_params_w_m(starting_params, key_M, D, K):
    spm = starting_params.get(key_M, {})
    w_mean = 1 * spm['w_mu'] if 'w_mu' in spm else np.random.normal(0, 1.0, size=(D, K))
    w_var  = 1 * spm['w_var'] if 'w_var' in spm else np.ones((D, K))
    return w_mean, w_var


# ------------------------ CLING-FA main ------------------------
class ClingFA_ablation2:
    def __init__(self, data: Dict[str, np.ndarray], N: int, M: int, K: int, D: List[int],
                 mgp_a1: float = 3.0, mgp_a2: float = 3.0, mgp_b1: float = 1.0, mgp_b2: float = 1.0,
                 ard_a_alpha: float = 3.0, ard_b_alpha: float = 1.0,
                 a_tau: float = 1e-3, b_tau: float = 1e-3,
                 center_data: Optional[List[bool]] = None, starting_params: Optional[dict] = None,
                 prune_mode: str = 'per_view', prune_threshold: float = 0.005, prune_min_views: int = 1):

        # --- pruning config ---
        self.prune_mode = prune_mode
        self.prune_threshold = float(prune_threshold)
        self.prune_min_views = int(prune_min_views)
        self._inactive_counts = np.zeros(K, dtype=int)
        self.prune_patience = 2
        self._no_prune_cycles = 0

        self.N, self.M, self.K, self.D = N, M, K, D
        self.mgp_a1, self.mgp_a2 = float(mgp_a1), float(mgp_a2)
        self.mgp_b1, self.mgp_b2 = float(mgp_b1), float(mgp_b2)
        self.ard_a_alpha, self.ard_b_alpha = float(ard_a_alpha), float(ard_b_alpha)
        self.a_tau, self.b_tau = float(a_tau), float(b_tau)

        general_params = {'N': self.N, 'K': self.K, 'D': self.D, 'M': self.M}
        if starting_params is None:
            starting_params = {}
        for m in range(M):
            starting_params.setdefault(f'M{m}', {})
        if center_data is None:
            center_data = starting_params.get('centering_data', [False for _ in range(M)])

        # --- normalize input data ---
        if isinstance(data, (list, tuple)):
            data = {f"M{m}": (np.asarray(v)) for m, v in enumerate(data)}
        elif not isinstance(data, dict):
            arr = np.asarray(data)
            data = {"M0": arr}
            
        if isinstance(center_data, bool):
            center_data = [center_data] * self.M
        elif len(center_data) != self.M:
            center_data = [bool(center_data[0])] * self.M

        # --- initialize Z ---
        z_mean, z_var = starting_params_z(starting_params, self.N, self.K)
        self.node_z = nodeFA_z(vi_mu=z_mean, vi_var=z_var, general_params=general_params)

        self.nodelist_y: List[nodeFA_y_m] = []
        self.nodelist_w: List[nodeFA_w_m] = []
        self.nodelist_alpha: List[nodeFA_alpha_m] = []
        self.nodelist_tau: List[nodeFA_tau_m] = []

        for m in range(self.M):
            key_m = f'M{m}'
            data_m = data[key_m]

            # y (NaN-safe centering if requested)
            if center_data[m]:
                feature_mean_m = np.nanmean(data_m, axis=0)
                node_y_m = nodeFA_y_m(data_m - feature_mean_m, m, general_params)
                node_y_m.data_mean = feature_mean_m
            else:
                node_y_m = nodeFA_y_m(data_m + 0.0, m, general_params)
                node_y_m.data_mean = None
            self.nodelist_y.append(node_y_m)

            # W
            w_mu, w_var = starting_params_w_m(starting_params, key_m, D[m], K)
            node_w_m = nodeFA_w_m(m=m, w_mu=w_mu, w_var=w_var, general_params=general_params)
            self.nodelist_w.append(node_w_m)

            # alpha
            node_alpha_m = nodeFA_alpha_m(a_alpha=self.ard_a_alpha, b_alpha=self.ard_b_alpha,
                                          m=m, general_params=general_params)
            self.nodelist_alpha.append(node_alpha_m)

            # tau
            node_tau_m = nodeFA_tau_m(self.a_tau, self.b_tau, m, general_params)
            self.nodelist_tau.append(node_tau_m)

        # ---- global delta/gamma ----
        totalD = int(np.sum(self.D))
        self.node_delta = nodeMGP_delta_global(a1=self.mgp_a1, b1=self.mgp_b1,
                                               a2=self.mgp_a2, b2=self.mgp_b2,
                                               K=self.K, totalD=totalD,
                                               general_params=general_params)

        self.elbo = 0.0
        self._wire_MB()

    # ----- wiring -----
    def _wire_MB(self):
        self.node_z.MB(self.nodelist_y, self.nodelist_w, self.nodelist_tau)
        for m in range(self.M):
            self.nodelist_y[m].MB(self.nodelist_w[m], self.nodelist_tau[m])
            self.nodelist_w[m].MB(self.nodelist_alpha[m], self.node_z,
                                  self.nodelist_y[m], self.nodelist_tau[m], self.node_delta)
            self.nodelist_alpha[m].MB(self.nodelist_w[m], self.node_delta)
            self.nodelist_tau[m].MB(self.nodelist_y[m], self.nodelist_w[m], self.node_z)
        self.node_delta.MB(self.nodelist_w, self.nodelist_alpha)

    # ----- core VI loop -----
    def _update_once(self):
        # 1) Z
        for k in range(self.K):
            self.node_z.update_k(k)
            for m in range(self.M):
                self.nodelist_w[m].update_params_z()
                self.nodelist_tau[m].update_params_w_z()

        # 2) W
        for k in range(self.K):
            for m in range(self.M):
                self.nodelist_w[m].update_k(k)
                self.nodelist_w[m].update_params_z()
                self.nodelist_tau[m].update_params_w_z()

        # 3) tau
        for m in range(self.M):
            self.nodelist_tau[m].update()
            self.nodelist_tau[m].update_params_w_z()

        # 4) alpha
        for m in range(self.M): 
            self.nodelist_alpha[m].update()

        # 5) global delta (MGP)
        self.node_delta.update_all()

    def _ELBO_once(self):
        self.node_z.ELBO()
        total = self.node_z.elbo
        for m in range(self.M):
            self.nodelist_w[m].ELBO()
            self.nodelist_alpha[m].ELBO()
            self.nodelist_y[m].ELBO()
            self.nodelist_tau[m].ELBO()
            total += ( self.nodelist_w[m].elbo + self.nodelist_alpha[m].elbo
                     + self.nodelist_y[m].elbo + self.nodelist_tau[m].elbo )
        self.node_delta.ELBO()
        total += self.node_delta.elbo
        self.elbo = total
        return total

    # ----- EV utilities -----
    def _get_Yraw_and_mu(self, m: int) -> Tuple[np.ndarray, np.ndarray]:
        Ym = self.nodelist_y[m].data
        mu = self.nodelist_y[m].data_mean
        if mu is None:
            Y_raw = Ym
            mu = np.nanmean(Y_raw, axis=0, keepdims=True)
        else:
            mu = mu.reshape(1, -1)
            Y_raw = Ym + mu
        return Y_raw, mu

    @staticmethod
    def _nansumsq(A: np.ndarray) -> float:
        return np.nansum(A * A)

    def variance_explained_per_view(self) -> np.ndarray:
        R2m = np.zeros(self.M)
        Z = self.node_z.E_z
        for m in range(self.M):
            Y_raw, mu = self._get_Yraw_and_mu(m)
            Yc = Y_raw - mu
            Msk = self.nodelist_y[m].M
            Wm = self.nodelist_w[m].E_w
            recon = Z @ Wm.T
            ss_res = self._nansumsq((Yc - recon) * Msk)
            ss_tot = self._nansumsq(Yc * Msk)
            R2m[m] = 1.0 - ss_res / (ss_tot + 1e-12) if ss_tot > 0 else 0.0
        return np.clip(R2m, 0.0, 1.0)

    def variance_explained_per_factor(self) -> np.ndarray:
        R2k = np.zeros(self.K)
        Z = self.node_z.E_z
        for k in range(self.K):
            gain_total, denom_total = 0.0, 0.0
            for m in range(self.M):
                Y_raw, mu = self._get_Yraw_and_mu(m)
                Yc  = Y_raw - mu
                Msk = self.nodelist_y[m].M
                Wm  = self.nodelist_w[m].E_w
                mu_all = Z @ Wm.T
                contrib = np.outer(Z[:, k], Wm[:, k])

                resid_with    = (Yc - mu_all) * Msk
                resid_without = (Yc - (mu_all - contrib)) * Msk

                ss_res        = self._nansumsq(resid_with)
                ss_res_wo     = self._nansumsq(resid_without)
                gain_total   += (ss_res_wo - ss_res)
                denom_total  += self._nansumsq(Yc * Msk)

            R2k[k] = gain_total / (denom_total + 1e-12) if denom_total > 0 else 0.0
        return np.clip(R2k, 0.0, 1.0)

    def variance_explained_per_factor_view(self) -> np.ndarray:
        R2_km = np.zeros((self.K, self.M))
        Z = self.node_z.E_z
        for m in range(self.M):
            Y_raw, mu = self._get_Yraw_and_mu(m)
            Yc  = Y_raw - mu
            Msk = self.nodelist_y[m].M
            Wm  = self.nodelist_w[m].E_w
            mu_all = Z @ Wm.T
            denom = self._nansumsq(Yc * Msk)
            if denom <= 0:
                continue
            for k in range(self.K):
                contrib = np.outer(Z[:, k], Wm[:, k])
                resid_with    = (Yc - mu_all) * Msk
                resid_without = (Yc - (mu_all - contrib)) * Msk
                gain = self._nansumsq(resid_without) - self._nansumsq(resid_with)
                R2_km[k, m] = gain / (denom + 1e-12)
        return np.clip(R2_km, 0.0, 1.0)

    def active_mask_by_R2(self, threshold: float = 0.02):
        R2_km = self.variance_explained_per_factor_view()
        active_km = R2_km >= threshold
        active_k = np.any(active_km, axis=1)
        return active_km, active_k, R2_km

    # ---- pruning ----
    def delete_inactive(self, tres: float = None, verbose: bool = True) -> bool:
        if tres is None:
            tres = self.prune_threshold
        mode = self.prune_mode
        min_views = self.prune_min_views

        if mode == 'per_view':
            R2_km = self.variance_explained_per_factor_view()
            is_active = (R2_km >= tres).sum(axis=1) >= min_views
        else:
            var_expl = self.variance_explained_per_factor()
            is_active = var_expl >= tres

        self._inactive_counts = np.where(is_active, 0, self._inactive_counts + 1)
        to_delete = self._inactive_counts >= self.prune_patience
        active = ~to_delete

        num_inactive = int(np.sum(to_delete))
        are_deleted = False

        if num_inactive > 0:
            self._apply_active_mask(active, verbose=verbose, msg="Deleted inactive")
            self._no_prune_cycles = 0
            are_deleted = True
            if verbose:
                print(f"[prune] Deleted {num_inactive} factors; K -> {self.K}")
        else:
            self._no_prune_cycles += 1

        return are_deleted

    # ---- adding ----
    def add_factors(self, k_new: int = 1, init_scale: float = 1e-2,
                    use_residual_pca: bool = True, verbose: bool = True):
        if k_new <= 0:
            return
        oldK = self.K
        N, M, D = self.N, self.M, self.D

        if use_residual_pca:
            z_mu_new = np.zeros((N, k_new))
            w_mu_news = []
            Z_old = self.node_z.E_z[:, :oldK] if oldK > 0 else np.zeros((N, 0))
            for m in range(M):
                Y_raw, mu = self._get_Yraw_and_mu(m)
                Yc = Y_raw - mu
                Wm_old = self.nodelist_w[m].E_w[:, :oldK] if oldK > 0 else np.zeros((D[m], 0))
                Rm = Yc - (Z_old @ Wm_old.T if oldK > 0 else 0.0)
                try:
                    u, s, vt = np.linalg.svd(np.nan_to_num(Rm), full_matrices=False)
                    z_mu_new[:, 0] += u[:, 0] * s[0] / max(1, M)
                    w_mu_news.append(vt.T[:, :k_new] * init_scale)
                except np.linalg.LinAlgError:
                    z_mu_new[:, 0] += np.random.normal(0, init_scale, size=N)
                    w_mu_news.append(np.random.normal(0.0, init_scale, size=(D[m], k_new)))
        else:
            z_mu_new = np.random.normal(0.0, init_scale, size=(N, k_new))
            w_mu_news = [np.random.normal(0.0, init_scale, size=(D[m], k_new)) for m in range(M)]

        newK = oldK + k_new

        # Z
        self.node_z.K = newK
        self.node_z.vi_mu  = np.hstack([self.node_z.vi_mu,  z_mu_new])
        self.node_z.vi_var = np.hstack([self.node_z.vi_var, np.ones((N, k_new))])
        self.node_z.update_params()

        for m in range(M):
            # W
            w = self.nodelist_w[m]
            w.K = newK
            w.vi_mu  = np.hstack([w.vi_mu,  w_mu_news[m]])
            w.vi_var = np.hstack([w.vi_var, np.ones((D[m], k_new))])
            w.update_params()

            # ALPHA 
            a = self.nodelist_alpha[m]
            a.K = newK
            vi_a_new = np.full(k_new, a.a_alpha + 0.5*a.Dm)
            vi_b_new = np.full(k_new, a.b_alpha + 1e-3)
            a.vi_a = np.hstack([a.vi_a, vi_a_new])
            a.vi_b = np.hstack([a.vi_b, vi_b_new])
            a.update_params()

            w.update_params_z()
            self.nodelist_tau[m].update_params_w_z()

        # DELTA
        self.node_delta.K = newK
        vi_a_new_d = np.full(k_new, self.node_delta.a2)
        vi_b_new_d = np.full(k_new, self.node_delta.b2)
        self.node_delta.vi_a = np.hstack([self.node_delta.vi_a, vi_a_new_d])
        self.node_delta.vi_b = np.hstack([self.node_delta.vi_b, vi_b_new_d])
        self.node_delta.totalD = int(np.sum(self.D))
        self.node_delta.update_params()

        self.K = newK
        self._inactive_counts = np.hstack([self._inactive_counts, np.zeros(k_new, dtype=int)])
        self._no_prune_cycles = 0

        if verbose:
            print(f"[add] Added {k_new} factor(s); K -> {self.K}")

        self._wire_MB()

    def needs_more_capacity(self, target_view_R2: float = 0.90, min_views_below: int = 1) -> bool:
        R2m = self.variance_explained_per_view()
        below = np.sum(R2m < target_view_R2)
        return bool(below >= min_views_below)

    # ----- factor slicing -----
    def _apply_active_mask(self, active: np.ndarray, verbose: bool = True, msg: str = ""):
        newK = int(np.sum(active))
        if newK == self.K:
            return
        # Z
        self.node_z.K = newK
        self.node_z.vi_mu  = self.node_z.vi_mu[:, active]
        self.node_z.vi_var = self.node_z.vi_var[:, active]
        self.node_z.update_params()
        # per view
        for m in range(self.M):
            w = self.nodelist_w[m]
            w.K = newK
            w.vi_mu  = w.vi_mu[:, active]
            w.vi_var = w.vi_var[:, active]
            w.update_params()
            w.update_params_z()

            a = self.nodelist_alpha[m]
            a.K = newK
            a.vi_a = a.vi_a[active]
            a.vi_b = a.vi_b[active]
            a.update_params()

            self.nodelist_tau[m].update_params_w_z()

        # global delta
        self.node_delta.K = newK
        self.node_delta.vi_a = self.node_delta.vi_a[active]
        self.node_delta.vi_b = self.node_delta.vi_b[active]
        self.node_delta.update_params()

        self.K = newK
        self._inactive_counts = self._inactive_counts[active]
        if verbose:
            print(f"[lock] {msg} → K -> {self.K}")
        self._wire_MB()

    def _force_set_K(self, newK: int, verbose: bool = True):
        if newK >= self.K:
            return
        try:
            scores = self.variance_explained_per_factor()
        except Exception:
            scores = np.array([sum((wm.E_w[:, k] ** 2).sum() for wm in self.nodelist_w)
                               for k in range(self.K)], dtype=float)
        keep_idx = np.argsort(-scores)[:newK]
        active = np.zeros(self.K, dtype=bool)
        active[keep_idx] = True
        self._apply_active_mask(active, verbose=verbose, msg=f"force-set K={newK}")

    # ---- two-value suffix helper ----
    @staticmethod
    def _two_value_suffix_min(seq: List[int], max_len: int = 12, min_len: int = 4) -> Tuple[bool, Optional[int]]:
        if not seq:
            return False, None
        suffix = []
        uniq = set()
        for x in reversed(seq[-max_len:]):
            if x not in uniq and len(uniq) == 2:
                break
            suffix.append(x)
            uniq.add(x)
        uniq = sorted(uniq)
        if len(uniq) == 2 and len(suffix) >= min_len:
            return True, uniq[0]
        return False, None

    # ----- public fit -----
    def fit(self, max_iter: Optional[int] = None, tol: float = 1e-3,
            prune_every: Optional[int] = 50,
            prune_warmup: int = 100,  
            add_every: Optional[int] = 50,
            add_patience: int = 2, add_k: int = 1,
            target_view_R2: float = 0.90, min_views_below: int = 1,
            max_K: Optional[int] = None, verbose: bool = True,
            two_value_lock: bool = True,
            two_value_history_len: int = 12,
            two_value_min_len: int = 4,
            lock_extra_iters: int = 50,
            oscillation_guard: bool = True,
            oscillation_flips: int = 5,
            oscillation_window: int = 12,
            no_capacity_change_patience: Optional[int] = 100):

        import itertools
        from tqdm.auto import tqdm

        elbos: List[float] = []
        K_list: List[int] = []
        egamma_list = []  # track global gamma
        last_elbo = None

        capacity_locked = False
        lock_iter = None
        last_K_for_osc = self.K
        last_dir = 0
        flips = 0
        recent_K_changes: List[int] = []

        iters_since_capacity_change = 0
        final_prune_done = False

        pbar = tqdm(total=max_iter if max_iter is not None else None,
                    desc="CLING-FA fit" + ("" if max_iter is not None else " (∞)"),
                    ncols=1000, disable=not verbose)

        it = 0
        for it in itertools.count():
            if (max_iter is not None) and (it >= max_iter):
                break

            self._update_once()
            elbo = self._ELBO_once()
            elbos.append(elbo); K_list.append(self.K)

            if verbose:
                try: pbar.set_postfix({"ELBO": f"{elbo:.1f}", "K": self.K})
                except Exception: pass
                pbar.update(1)

            if it % 10 == 0:
                egamma_list.append(self.node_delta.E_gamma.copy())

            prev_K = self.K 

            # ---- PRUNE step ----
            if (not capacity_locked) and prune_every and it > 0 and (it % prune_every == 0) and (it >= prune_warmup):
                did_prune = self.delete_inactive(verbose=verbose)
                if did_prune:
                    self._wire_MB()
            elif (not capacity_locked) and prune_every and it > 0 and (it % prune_every == 0) and (it < prune_warmup):
                self._no_prune_cycles += 1

            if (not capacity_locked) and add_every and it > 0 and (it % add_every == 0):
                if self._no_prune_cycles >= max(1, add_patience):
                    if self.needs_more_capacity(target_view_R2, min_views_below):
                        if (max_K is None) or (self.K + add_k <= max_K):
                            self.add_factors(k_new=add_k, verbose=verbose)

            # ---- Oscillation bookkeeping ----
            if self.K != last_K_for_osc:
                direction = int(np.sign(self.K - last_K_for_osc))
                if last_dir != 0 and direction != 0 and direction != last_dir:
                    flips += 1
                last_dir = direction if direction != 0 else last_dir
                last_K_for_osc = self.K
                recent_K_changes.append(self.K)
                if len(recent_K_changes) > max(two_value_history_len, oscillation_window):
                    recent_K_changes = recent_K_changes[-max(two_value_history_len, oscillation_window):]

            # ---- No-capacity-change early stop ----
            if self.K != prev_K:
                iters_since_capacity_change = 0
            else:
                iters_since_capacity_change += 1
                if (not capacity_locked) and (no_capacity_change_patience is not None) \
                   and (iters_since_capacity_change >= no_capacity_change_patience):
                    if verbose:
                        print(f"\nStopped: no add/delete for {no_capacity_change_patience} consecutive iterations.")
                    break

            # ---- Two-value lock ----
            if two_value_lock and (not capacity_locked):
                ok, k_target = self._two_value_suffix_min(
                    recent_K_changes,
                    max_len=two_value_history_len,
                    min_len=two_value_min_len
                )
                if ok and (k_target < self.K):
                    self._force_set_K(k_target, verbose=True)
                    capacity_locked = True
                    lock_iter = it
                    if verbose:
                        print(f"[lock] Two-value oscillation detected; "
                              f"freezing capacity at K={self.K} and running {lock_extra_iters} more iterations.")

            # ---- Generic guard ----
            if (not capacity_locked) and oscillation_guard and flips >= oscillation_flips and len(recent_K_changes) > 0:
                k_target = int(np.percentile(recent_K_changes[-oscillation_window:], 25))
                if k_target < self.K:
                    self._force_set_K(k_target, verbose=True)
                capacity_locked = True
                lock_iter = it
                if verbose:
                    print(f"[lock] Oscillation detected (flips={flips}); "
                          f"freezing capacity at K={self.K} and running {lock_extra_iters} more iterations.")

            if (not final_prune_done) and capacity_locked and lock_iter is not None and (it - lock_iter) == 0:
                self.delete_inactive(verbose=verbose)
                self._wire_MB()
                final_prune_done = True

            if (not final_prune_done) and max_iter is not None and (max_iter - it) == 50:
                if verbose: 
                    print("[final prune] running last cleanup")
                self.delete_inactive(verbose=verbose)
                self._wire_MB()
                final_prune_done = True
                capacity_locked = True 
                lock_iter = it 

            # ---- Convergence / stopping ----
            if capacity_locked and lock_iter is not None and (it - lock_iter) >= lock_extra_iters:
                if verbose: print(f"\nStopped after lock: ran {lock_extra_iters} extra iterations.")
                break

            if (not capacity_locked) and last_elbo is not None and \
               abs(elbo - last_elbo) / (abs(last_elbo) + 1e-12) < tol:
                if verbose: print(f"\nConverged at iter {it+1}")
                break
            last_elbo = elbo

        if verbose:
            pbar.close()

        return elbos, K_list, egamma_list

    # ----- API -----
    def transform(self) -> np.ndarray:
        return self.node_z.E_z

    def get_factors(self) -> np.ndarray:
        return self.transform()

    def get_weights(self, view: Optional[int] = None):
        if view is None:
            return [wm.E_w for wm in self.nodelist_w]
        return self.nodelist_w[view].E_w

    def reconstruct(self, view: Optional[int] = None) -> List[np.ndarray] | np.ndarray:
        Z = self.node_z.E_z
        if view is None:
            return [Z @ wm.E_w.T for wm in self.nodelist_w]
        return Z @ self.nodelist_w[view].E_w.T

    # ---- plotting ----
    def plot_variance_explained(self, per_factor: bool = True, per_view: bool = True):
        import numpy as np
        if go is None:
            import matplotlib.pyplot as plt

            if per_view:
                R2m = self.variance_explained_per_view()
                plt.figure(figsize=(6, 4))
                plt.bar(np.arange(self.M), R2m)
                plt.xlabel("View"); plt.ylabel("R^2"); plt.title("Variance explained per view")
                plt.show()

            if per_factor:
                R2km = self.variance_explained_per_factor_view()
                plt.figure(figsize=(8, 5))
                bottom = np.zeros(self.K)
                for m in range(self.M):
                    plt.bar(np.arange(self.K), R2km[:, m], bottom=bottom, label=f"View {m}")
                    bottom += R2km[:, m]
                plt.xlabel("Factor"); plt.ylabel("Variance explained (stacked)")
                plt.title("Variance explained per factor (by view)"); plt.legend(); plt.show()
        else:
            fig = go.Figure()
            if per_view:
                R2m = self.variance_explained_per_view()
                fig.add_trace(go.Bar(x=[f"View {m}" for m in range(self.M)], y=R2m, name="Per view"))
            if per_factor:
                R2km = self.variance_explained_per_factor_view()
                for m in range(self.M):
                    fig.add_trace(go.Bar(x=[f"F{k}" for k in range(self.K)], y=R2km[:, m], name=f"View {m}"))
            fig.update_layout(barmode='stack', title="Variance explained",
                              xaxis_title="Factors/Views", yaxis_title="R^2")
            fig.show()

    def plot_diagnostics(self, elbos, K_list, log_scale: bool = False):
        iters = np.arange(len(elbos))
        if go is None:
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(figsize=(8, 4))
            color = 'tab:blue'
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("ELBO", color=color)
            ax1.plot(iters, elbos, color=color)
            if log_scale: ax1.set_yscale("log")
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel("K (factors)", color=color)
            ax2.plot(iters, K_list, color=color, linestyle="--")
            ax2.tick_params(axis='y', labelcolor=color)
            plt.title("Training diagnostics"); plt.tight_layout(); plt.show()
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=iters, y=elbos, mode="lines", name="ELBO", yaxis="y1"))
            fig.add_trace(go.Scatter(x=iters, y=K_list, mode="lines+markers", name="K (factors)", yaxis="y2"))
            fig.update_layout(
                title="Training diagnostics",
                xaxis=dict(title="Iteration"),
                yaxis=dict(title="ELBO", side="left", type="log" if log_scale else "linear"),
                yaxis2=dict(title="K (factors)", overlaying="y", side="right"),
                legend=dict(x=0.01, y=0.99), width=900, height=500
            )
            fig.show()

    # ----- builder -----
    @staticmethod
    def from_numpy_views(views, K=None, center=None,
                         initK_mode="paper", initK_c=5.0,
                         init_mode="random",
                         mgp_a1=3.0, mgp_a2=3.0, mgp_b1=1.0, mgp_b2=1.0,
                         ard_a_alpha=0.5, ard_b_alpha=1.0,
                         a_tau=1e-3, b_tau=1e-3,
                         starting_params=None,
                         prune_mode='per_view',
                         prune_threshold=0.005,
                         prune_min_views=1):
        views = views if isinstance(views, list) else [views[k] for k in sorted(views.keys())]
        M = len(views)
        N = views[0].shape[0]
        D = [v.shape[1] for v in views]
        if K is None:
            K = compute_init_K(D, M, mode=initK_mode, c=initK_c)

        data = {f"M{m}": views[m] for m in range(M)}

        # PCA init
        if init_mode == "pca":
            Y_concat = []
            for m, v in enumerate(views):
                Yc = v - np.nanmean(v, axis=0, keepdims=True)
                Y_concat.append(np.nan_to_num(Yc))
            Y_all = np.hstack(Y_concat)
            U, S, Vt = np.linalg.svd(Y_all, full_matrices=False)
            Z_init = U[:, :K] * S[:K]
            W_inits = []
            start = 0
            for m, d in enumerate(D):
                Wm = Vt[:K, start:start + d].T
                W_inits.append(Wm)
                start += d
            starting_params = starting_params or {}
            starting_params['z_mu'] = Z_init
            starting_params['z_var'] = np.ones((N, K))
            for m in range(M):
                starting_params[f"M{m}"] = {
                    'w_mu': W_inits[m],
                    'w_var': np.ones((D[m], K))
                }

        return ClingFA_ablation2(data=data, N=N, M=M, K=K, D=D,
                       mgp_a1=mgp_a1, mgp_a2=mgp_a2, mgp_b1=mgp_b1, mgp_b2=mgp_b2,
                       ard_a_alpha=ard_a_alpha, ard_b_alpha=ard_b_alpha,
                       a_tau=a_tau, b_tau=b_tau,
                       center_data=center, starting_params=starting_params,
                       prune_mode=prune_mode,
                       prune_threshold=prune_threshold,
                       prune_min_views=prune_min_views)
