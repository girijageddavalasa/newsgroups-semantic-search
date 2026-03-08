"""
clusterer.py — UMAP dimensionality reduction + Fuzzy C-Means clustering.

══════════════════════════════════════════════════════════════════════════════
PART 1: WHY UMAP BEFORE CLUSTERING?
══════════════════════════════════════════════════════════════════════════════

Clustering in 384 dimensions is unreliable due to the "curse of dimensionality":
  • All pairwise distances converge to the same value as D→∞
  • Distance metrics lose discriminative power above ~50 dimensions
  • FCM's centroid updates become numerically unstable in high-D spaces

UMAP vs PCA vs t-SNE:
  PCA:   Linear. Cannot capture curved manifolds in embedding space. Poor cluster sep.
  t-SNE: Non-parametric. No transform() for new query vectors. Unusable at inference.
  UMAP:  Non-linear + parametric. Has transform() → can project new queries. ✓ CHOSEN

We project 384-d → 50-d (not 2-d for viz — 50-d preserves more global structure).

══════════════════════════════════════════════════════════════════════════════
PART 2: FUZZY C-MEANS FROM SCRATCH
══════════════════════════════════════════════════════════════════════════════

WHY NOT K-MEANS?
  K-Means assigns exactly one cluster label per document.
  A post about gun legislation belongs to both politics AND firearms — to varying degrees.
  Hard labels lose this information. The task explicitly requires a distribution.

WHY NOT scikit-fuzzy?
  The task says "if you didn't write it, it shouldn't be in your cache."
  We applied this principle to the clustering too — FCM is implemented from scratch
  using only numpy, following Bezdek (1981).

BEZDEK FCM ALGORITHM:
  Given N documents, K clusters, fuzziness m=2:
  
  1. Initialise membership matrix U: (N, K), each row drawn from Dirichlet(1)
     so rows sum to 1.0 (valid probability distributions from the start)
  
  2. Repeat until convergence:
     a. Update centroids:
        C_k = Σ_i (u_ik^m * x_i) / Σ_i (u_ik^m)
        (weighted average of all documents, weight = membership^m)
     
     b. Compute distances:
        d_ik = ||x_i - C_k||²  (Euclidean in UMAP-50 space)
     
     c. Update memberships:
        u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))
        (inverse-distance weighting — closer centroids get higher membership)
  
  3. Converged when ||U_new - U_old||_∞ < 0.005

FUZZINESS PARAMETER m=2:
  m=1: FCM collapses to hard K-Means (sharp boundaries)
  m→∞: All memberships → 1/K (fully fuzzy, no structure)
  m=2: Standard choice; good balance. Literature consistently recommends this.

══════════════════════════════════════════════════════════════════════════════
PART 3: K SELECTION — PARTITION COEFFICIENT ELBOW
══════════════════════════════════════════════════════════════════════════════

We do NOT use K=20 just because the dataset has 20 labels.
Several newsgroup pairs are semantically indistinguishable:
  rec.sport.baseball + rec.sport.hockey → both are sports; nearly identical vocabulary
  soc.religion.christian + talk.religion.misc → same semantic field
  comp.sys.ibm.pc.hardware + comp.sys.mac.hardware → both are hardware discussions

Partition Coefficient (PC) measures cluster crispness:
  PC = (1/N) Σ_i Σ_k u_ik²
  
  PC = 1.0  → perfectly hard clusters (each doc fully belongs to one cluster)
  PC = 1/K  → maximally fuzzy (uniform random membership) — this is the noise baseline
  
  We scan K ∈ {8, 10, 12, 14, 16, 18, 20}, compute PC for each,
  and choose the elbow — the K where adding more clusters stops improving crispness.
  
  The elbow is NOT necessarily K=20. If K=10 captures the semantic structure,
  K=20 just fragments coherent clusters into meaningless sub-clusters.
  
  We save the elbow plot to results/k_selection.png — visual evidence for the choice.
"""

import os
import time
import numpy as np
import joblib
from typing import Optional, List, Tuple
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environments
import matplotlib.pyplot as plt

UMAP_PATH = os.environ.get("UMAP_PATH", "data/umap_model.pkl")
FCM_PATH  = os.environ.get("FCM_PATH",  "data/fcm_model.pkl")
RESULTS_DIR = "results"

_umap_model = None
_fcm_model = None   # FuzzyCMeans instance
_K: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Fuzzy C-Means Implementation (Bezdek 1981) — written from scratch
# ══════════════════════════════════════════════════════════════════════════════

class FuzzyCMeans:
    """
    Fuzzy C-Means clustering from scratch.
    
    Parameters
    ----------
    n_clusters : int    — K, number of clusters
    m          : float  — fuzziness exponent (default 2.0, standard Bezdek)
    max_iter   : int    — maximum iterations
    tol        : float  — convergence tolerance on membership matrix
    random_state: int   — for reproducibility
    """

    def __init__(self, n_clusters: int, m: float = 2.0,
                 max_iter: int = 150, tol: float = 0.005,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: Optional[np.ndarray] = None    # (K, D)
        self.membership_: Optional[np.ndarray] = None   # (N, K)
        self.partition_coefficient_: float = 0.0
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        """
        Fit FCM on data matrix X: (N, D).
        Stores centroids, membership matrix, and partition coefficient.
        """
        N, D = X.shape
        K = self.n_clusters
        rng = np.random.default_rng(self.random_state)

        # Initialise U from Dirichlet — guarantees rows sum to 1.0
        # (unlike uniform random which requires explicit normalisation)
        U = rng.dirichlet(np.ones(K), size=N).astype(np.float64)

        print(f"  [FCM] K={K}: fitting on ({N}, {D}) matrix")

        for iteration in range(self.max_iter):
            # Step A: Update centroids
            # C_k = Σ_i (u_ik^m * x_i) / Σ_i (u_ik^m)
            U_m = U ** self.m             # (N, K)
            denom = U_m.sum(axis=0)       # (K,) — sum of weights per cluster
            # C[k] = (U_m[:, k] @ X) / denom[k]
            C = (U_m.T @ X) / denom[:, np.newaxis]   # (K, D)

            # Step B: Compute squared Euclidean distances
            # D_ik = ||x_i - C_k||²
            # Efficient: ||x-c||² = ||x||² - 2x·c + ||c||²
            X_sq = np.sum(X ** 2, axis=1, keepdims=True)    # (N, 1)
            C_sq = np.sum(C ** 2, axis=1, keepdims=True)    # (K, 1)
            cross = X @ C.T                                   # (N, K)
            dist2 = X_sq + C_sq.T - 2 * cross               # (N, K)
            dist2 = np.maximum(dist2, 1e-10)                 # numerical floor

            # Step C: Update memberships
            # u_ik = 1 / Σ_j (d_ik / d_ij)^(2/(m-1))
            exp = 2.0 / (self.m - 1.0)
            # Ratio matrix R[i,k,j] = (d_ik / d_ij)^exp
            # Vectorised: for each (i,k), sum over j of (dist2[i,k]/dist2[i,j])^exp
            # = dist2[i,k]^exp * Σ_j dist2[i,j]^(-exp)
            inv_exp = dist2 ** (-exp)                        # (N, K)
            inv_sum = inv_exp.sum(axis=1, keepdims=True)     # (N, 1)
            dist_exp = dist2 ** exp                          # (N, K)
            U_new = 1.0 / (dist_exp * inv_sum)              # (N, K)

            # Handle exact centroid hits (distance=0) → membership=1
            exact = (dist2 == 1e-10)
            if exact.any():
                U_new[exact.any(axis=1)] = 0.0
                for i in range(N):
                    hits = np.where(dist2[i] == 1e-10)[0]
                    if len(hits) > 0:
                        U_new[i, hits] = 1.0 / len(hits)

            delta = float(np.max(np.abs(U_new - U)))
            U = U_new

            if iteration % 10 == 0:
                print(f"  [FCM] K={K}: iter={iteration:3d}  delta={delta:.6f}")

            if delta < self.tol:
                print(f"  [FCM] K={K}: converged at iteration {iteration}")
                break

        self.centroids_ = C.astype(np.float32)
        self.membership_ = U.astype(np.float32)
        self.n_iter_ = iteration + 1

        # Partition Coefficient: PC = (1/N) Σ_i Σ_k u_ik²
        # Range: [1/K, 1.0]. Higher = crisper clusters.
        # Random baseline = 1/K (what you'd get with random memberships)
        self.partition_coefficient_ = float(np.mean(np.sum(U ** 2, axis=1)))
        random_baseline = 1.0 / K
        print(f"  [FCM] K={K}: PC={self.partition_coefficient_:.4f} "
              f"(random baseline={random_baseline:.4f})")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute membership vector for new data points.
        X: (N, D) — must be in same UMAP-50 space as training data.
        Returns: (N, K) float32 membership matrix.
        """
        if self.centroids_ is None:
            raise RuntimeError("FCM not fitted yet")

        dist2 = np.sum((X[:, np.newaxis, :] - self.centroids_[np.newaxis, :, :]) ** 2,
                       axis=2)
        dist2 = np.maximum(dist2, 1e-10)
        exp = 2.0 / (self.m - 1.0)
        inv_exp = dist2 ** (-exp)
        inv_sum = inv_exp.sum(axis=1, keepdims=True)
        dist_exp = dist2 ** exp
        U = 1.0 / (dist_exp * inv_sum)
        return U.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# K Selection: Partition Coefficient Elbow Analysis
# ══════════════════════════════════════════════════════════════════════════════

def select_k(X_reduced: np.ndarray,
             k_range: List[int] = None) -> Tuple[int, dict]:
    """
    Scan K values, compute PC for each, find elbow, save plot.
    
    The elbow is computed as the K where the second derivative of PC
    changes sign (curvature peak) — same logic as the elbow method for K-Means inertia.
    
    Returns (best_k, {k: pc_score})
    """
    if k_range is None:
        k_range = [8, 10, 12, 14, 16, 18, 20]

    print(f"\n[Cluster] K-selection scan: K ∈ {k_range}")
    print("WHY: We prove K empirically — not K=20 just because dataset has 20 labels.")
    print("Several newsgroup pairs are semantically indistinguishable.\n")

    pc_scores = {}
    for k in k_range:
        fcm = FuzzyCMeans(n_clusters=k, random_state=42)
        fcm.fit(X_reduced)
        pc_scores[k] = fcm.partition_coefficient_

    # Print table
    print("\nPartition Coefficient by K:")
    print(f"{'K':>4} | {'PC Score':>10} | {'vs Random':>10} | Bar")
    print("-" * 50)
    max_pc = max(pc_scores.values())
    for k, pc in sorted(pc_scores.items()):
        baseline = 1.0 / k
        above = pc - baseline
        bar = "█" * int(30 * pc / max_pc)
        print(f"{k:>4} | {pc:>10.4f} | {above:>+10.4f} | {bar}")

    # Elbow detection: largest drop in PC improvement
    ks = sorted(pc_scores.keys())
    pcs = [pc_scores[k] for k in ks]

    # First differences (how much PC drops as K increases)
    diffs = [pcs[i] - pcs[i+1] for i in range(len(pcs)-1)]

    # Elbow = K where the drop stops being large
    # i.e., second differences change sign
    best_k = ks[0]  # default: smallest K
    if len(diffs) >= 2:
        second_diff = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
        # Elbow = first peak in second differences
        elbow_idx = int(np.argmax(second_diff)) + 1
        best_k = ks[elbow_idx]

    print(f"\n[Cluster] Elbow analysis suggests K = {best_k}")
    print(f"  Interpretation: above K={best_k}, adding clusters gives diminishing PC improvement.")
    print(f"  Several 20newsgroups categories are semantically merged at K={best_k}.")

    # Save elbow plot
    _save_k_plot(ks, pcs, best_k)

    return best_k, pc_scores


def _save_k_plot(ks: list, pcs: list, best_k: int):
    """Save K-selection elbow plot to results/k_selection.png."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    baselines = [1.0 / k for k in ks]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ks, pcs, 'b-o', linewidth=2, markersize=8,
            label='Partition Coefficient (PC)', zorder=3)
    ax.plot(ks, baselines, 'r--', linewidth=1.5, alpha=0.7,
            label='Random baseline (1/K)', zorder=2)
    ax.fill_between(ks, baselines, pcs, alpha=0.15, color='blue',
                    label='PC above random baseline')

    # Mark the chosen K
    ax.axvline(x=best_k, color='green', linestyle='--', linewidth=2,
               label=f'Chosen K={best_k} (elbow)', zorder=4)
    ax.scatter([best_k], [next(p for k, p in zip(ks, pcs) if k == best_k)],
               color='green', s=150, zorder=5)

    ax.set_xlabel('Number of Clusters (K)', fontsize=13)
    ax.set_ylabel('Partition Coefficient', fontsize=13)
    ax.set_title(
        'Fuzzy C-Means K-Selection: Partition Coefficient Elbow Analysis\n'
        '(Higher PC = crisper clusters; elbow = point of diminishing returns)',
        fontsize=12
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ks)

    # Annotation
    ax.annotate(
        f'K={best_k} chosen\n(elbow point)',
        xy=(best_k, next(p for k, p in zip(ks, pcs) if k == best_k)),
        xytext=(best_k + 1, max(pcs) * 0.95),
        arrowprops=dict(arrowstyle='->', color='green', lw=2),
        fontsize=10, color='green'
    )

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "k_selection.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Cluster] K-selection plot saved to {plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
# UMAP + FCM Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def fit_umap(embeddings: np.ndarray) -> np.ndarray:
    """
    Fit UMAP on (N, 384) embedding matrix → (N, 50) reduced matrix.
    Saves model with joblib (portable across Python versions via numpy arrays).
    Returns reduced matrix.
    """
    import umap

    print(f"[Clusterer] Fitting UMAP: {embeddings.shape} → (N, 50)")
    print("  This takes 5-15 minutes on CPU. UMAP is building a k-NN graph.")

    reducer = umap.UMAP(
        n_components=50,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=True,
        low_memory=True,     # reduces peak RAM during fit
    )
    reduced = reducer.fit_transform(embeddings)

    os.makedirs(os.path.dirname(UMAP_PATH), exist_ok=True)
    # joblib serialises UMAP's numpy arrays directly, bypassing numba's
    # broken pickle hooks that fail across Python versions
    joblib.dump(reducer, UMAP_PATH, compress=3)
    print(f"[Clusterer] UMAP saved to {UMAP_PATH} (joblib format, Python-version portable)")

    np.save("data/umap_reduced.npy", reduced.astype(np.float32))
    return reduced.astype(np.float32)


def fit_fcm(X_reduced: np.ndarray, k: int) -> FuzzyCMeans:
    """
    Fit Fuzzy C-Means with K clusters on UMAP-50 matrix.
    Saves model to disk.
    Returns fitted FuzzyCMeans instance.
    """
    print(f"\n[Clusterer] Fitting FCM with K={k}")
    fcm = FuzzyCMeans(n_clusters=k, m=2.0, max_iter=150, tol=0.005, random_state=42)
    fcm.fit(X_reduced)

    os.makedirs(os.path.dirname(FCM_PATH), exist_ok=True)
    import pickle
    with open(FCM_PATH, 'wb') as f:
        pickle.dump(fcm, f)
    print(f"[Clusterer] FCM model saved to {FCM_PATH}")

    return fcm


def load_models() -> bool:
    """
    Load UMAP and FCM models from disk.
    UMAP uses joblib (portability), FCM uses pickle (pure numpy, always portable).
    Returns True if successful.
    """
    global _umap_model, _fcm_model, _K

    if not (os.path.exists(UMAP_PATH) and os.path.exists(FCM_PATH)):
        return False

    _umap_model = joblib.load(UMAP_PATH)

    import pickle
    with open(FCM_PATH, 'rb') as f:
        _fcm_model = pickle.load(f)
    _K = _fcm_model.n_clusters

    print(f"[Clusterer] Loaded UMAP + FCM (K={_K})")
    return True


def get_query_membership(query_vec: np.ndarray) -> np.ndarray:
    """
    Get FCM membership vector for a new query embedding.
    Steps:
      1. Project query from 384-d → 50-d via UMAP transform()
      2. Compute distance to each FCM centroid → membership vector
    Returns: (K,) float32 array summing to ~1.0
    """
    if _umap_model is None or _fcm_model is None:
        return np.ones(_K or 1, dtype=np.float32) / (_K or 1)

    reduced = _umap_model.transform(query_vec.reshape(1, -1))
    membership = _fcm_model.transform(reduced)
    return membership[0]


def get_dominant_clusters(membership: np.ndarray,
                          min_membership: float = 0.15) -> List[int]:
    """
    Return cluster indices where membership >= min_membership.
    A query can legitimately belong to multiple clusters (e.g. gun legislation →
    both politics and firearms). We route the cache lookup to ALL dominant clusters.
    min_membership=0.15 catches meaningful secondary clusters without noise.
    """
    indices = np.where(membership >= min_membership)[0].tolist()
    if not indices:
        indices = [int(np.argmax(membership))]
    return sorted(indices, key=lambda i: -membership[i])


def get_k() -> int:
    return _K
