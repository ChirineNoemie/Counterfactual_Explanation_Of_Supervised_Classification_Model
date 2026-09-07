"""
CounterfactualLVQ — counterfactuals guided by LVQ prototypes.

The prototype learning algorithm is LVQ1 (Algorithm 1).

Organized in two phases
───────────────────────

OFFLINE phase (in __init__, identical for binary and multiclass)
  LVQ1 over the whole Dtrain: nprot prototypes per class, labeled.
  For each (x, y): w* = argmin ||x − w||² over all classes combined,
  attraction if class(w*) = y, repulsion otherwise.
  Prototype learning is independent of the classifier:
  it only uses the data and their true labels.

ONLINE phase (compute_counterfactual)
  1. y': target class provided by the pipeline
  2. P[y'] = lvq_prototypes[y']          (already computed)
  3. line search for each prototype x_c:
       L(λ) = (1−λ)² + β·(1 − P(y'|λ·x0 + (1−λ)·x_c))²
     (1−λ)²: proximity to x0 (λ=1 ↔ x0, λ=0 ↔ prototype)
     (1−P)²: validity, P provided by the target class classifier
  4. validity: controlled by that same classifier f_y'
     (binary: clf.predict; multiclass: clf[y'].predict == 1)
"""

import time
import numpy as np
import scipy.optimize as opt


_METHOD_NAME = 'LVQ'


class CounterfactualLVQ:

    def __init__(self, clf, X_train, y_train, num_iter=100, beta=10.0,
                 n_prototypes=32, eps=1e-3):
        self.clf = clf
        self.num_iter = num_iter
        self.beta = beta
        self.n_prototypes = n_prototypes
        self.eps = eps
        self.n_clusters = n_prototypes
        self.cluster_method = 'lvq'
        self.generation_times = []
        self.target_class = None
        self.lvq_prototypes = {}

        self.X_train = X_train
        self.y_train = y_train

        # OFFLINE phase: LVQ1 over the whole Dtrain, regardless of the
        # classifier (binary or OVR dict). Prototype learning does not
        # depend on how the classifier was trained.
        if X_train is not None and y_train is not None:
            self._fit_lvq(X_train, y_train, n_prototypes)

    # ------------------------------------------------------------------
    # LVQ1 — Algorithm 1 (whole Dtrain, all classes)
    # ------------------------------------------------------------------

    def _fit_lvq(self, X_train, y_train, n_prototypes,
                 max_iter=2500, gtol=1e-5, eta=0.1):
        """
        LVQ1 over the whole Dtrain (all classes).

        W = nprot prototypes per class, each labeled.
        For each (x, y):
          w* = argmin_{w in W} ||x - w||²   (over all classes combined)
          if class(w*) == y  →  w* ← w* + η(x − w*)   (attraction)
          else               →  w* ← w* − η(x − w*)   (repulsion)
        """
        classes = np.unique(y_train)
        W_list, L_list = [], []
        for cls in classes:
            D_c = X_train[y_train == cls]
            n = min(n_prototypes, len(D_c))
            idx = np.random.choice(len(D_c), n, replace=False)
            W_list.append(D_c[idx].copy().astype(np.float64))
            L_list.extend([cls] * n)

        W = np.vstack(W_list)
        labels = np.array(L_list)

        for _ in range(max_iter):
            prev = W.copy()
            for i in np.random.permutation(len(X_train)):
                x, y = X_train[i], y_train[i]
                w_idx = np.argmin(np.sum((W - x) ** 2, axis=1))
                if labels[w_idx] == y:
                    W[w_idx] += eta * (x - W[w_idx])   # attraction
                else:
                    W[w_idx] -= eta * (x - W[w_idx])   # repulsion
            if np.max(np.linalg.norm(W - prev, axis=1)) < gtol:
                break

        for cls in classes:
            self.lvq_prototypes[cls] = W[labels == cls]

    # ------------------------------------------------------------------
    # Classifier querying (f_y' throughout)
    # ------------------------------------------------------------------

    def _get_target_proba(self, x, target_class):
        """
        P(y' | x).
        Binary      : clf.predict_proba(x)[index of y']
        Multiclass  : clf[y'].predict_proba(x)[1]  = f_y'(x)
        """
        x = np.array(x).reshape(1, -1)
        if isinstance(self.clf, dict):
            return self.clf[target_class].predict_proba(x)[0][1]
        classes = list(self.clf.classes_)
        return self.clf.predict_proba(x)[0][classes.index(target_class)]

    def _predict_class(self, x_c, target_class=None):
        """
        Binary      : clf.predict(x_c)
        Multiclass  : prediction from the target class classifier
                       f_y' — the same one that guides the optimization.
                       Returns y' if f_y' recognizes x_c, None otherwise.
        """
        x_c = np.array(x_c).reshape(1, -1)
        if isinstance(self.clf, dict):
            if self.clf[target_class].predict(x_c)[0] == 1:
                return target_class
            return None
        return self.clf.predict(x_c)[0]

    def _is_valid(self, x_c, target_class):
        return self._predict_class(x_c, target_class) == target_class

    # ------------------------------------------------------------------
    # Line search (identical for binary and multiclass)
    # ------------------------------------------------------------------

    def _line_search(self, x, x_c, target_class):
        """
        Minimizes L(λ) = (1−λ)² + β · (1 − P(y' | λ·x + (1−λ)·x_c))²
        via bounded Brent's method on [0, 1].

        λ = 1 ↔ x0 (instance), λ = 0 ↔ x_c (prototype).
        (1−λ)² penalizes distance from x0; (1−P)² penalizes the
        lack of confidence in the target class: the two terms
        are antagonistic, so the minimum settles near the boundary,
        on the target class side.
        """
        def objective(lam):
            # lam ∈ [0, 1]: 1 = x0, 0 = prototype
            x_interp = lam * x + (1 - lam) * x_c
            proba = self._get_target_proba(x_interp, target_class)
            return (1 - lam) ** 2 + self.beta * (1 - proba) ** 2

        best_l = opt.minimize_scalar(
            objective, bounds=(0, 1), method='bounded',
            options={'maxiter': self.num_iter}
        ).x
        return best_l * x + (1 - best_l) * x_c

    # ------------------------------------------------------------------
    # Generating candidates from the prototypes of y'
    # ------------------------------------------------------------------

    def _compute_candidates(self, x, target_class):
        P_target = self.lvq_prototypes.get(target_class)
        if P_target is None or len(P_target) == 0:
            return None
        return np.array([self._line_search(x, x_c, target_class)
                         for x_c in P_target])

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    def _single(self, x, target_class):
        candidates = self._compute_candidates(x, target_class)
        if candidates is None:
            return None
        for c in sorted(candidates, key=lambda c: np.linalg.norm(c - x)):
            if self._is_valid(c, target_class):
                return c
        return None

    def _greedy(self, x, target_class, N):
        candidates = self._compute_candidates(x, target_class)
        if candidates is None:
            return []
        sorted_cands = sorted(candidates, key=lambda c: np.linalg.norm(c - x))
        return [c for c in sorted_cands if self._is_valid(c, target_class)][:N]

    def _random(self, x, target_class, N):
        candidates = self._compute_candidates(x, target_class)
        if candidates is None:
            return []
        return [c for c in np.random.permutation(candidates)
                if self._is_valid(c, target_class)][:N]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def compute_counterfactual(self, x, target, N=1, model_override=None,
                               target_class=None, strategy='greedy', **kwargs):
        """
        The prototypes are already ready (offline phase, in __init__).
        Here: line search from each prototype of y', then
        selection of the N best valid candidates.
        """
        start = time.time()
        tc = target_class if target_class is not None else target
        self.target_class = tc

        if N == 1:
            result = self._single(x, tc)
        elif strategy == 'random':
            results = self._random(x, tc, N)
            result = np.array(results).reshape(len(results), -1) if results else None
        else:
            results = self._greedy(x, tc, N)
            result = np.array(results).reshape(len(results), -1) if results else None

        self.generation_times.append(time.time() - start)
        return result

    def log_generation_time(self, t):
        self.generation_times.append(t)

    def get_avg_generation_time(self):
        return np.mean(self.generation_times)

    @classmethod
    def get_class_name(cls):
        return _METHOD_NAME
