"""
CounterfactualLVQ — contrefactuels guidés par des prototypes LVQ.

L'algorithme d'apprentissage des prototypes est LVQ1 (Algorithme 1).

Organisation en deux phases
───────────────────────────

Phase OFFLINE (à l'__init__, identique binaire et multiclasse)
  LVQ1 sur tout Dtrain : nprot prototypes par classe, étiquetés.
  Pour chaque (x, y) : w* = argmin ||x − w||² toutes classes confondues,
  attraction si classe(w*) = y, répulsion sinon.
  L'apprentissage des prototypes est indépendant du classifieur :
  il n'utilise que les données et leurs vraies étiquettes.

Phase ONLINE (compute_counterfactual)
  1. y' : classe cible fournie par le pipeline
  2. P[y'] = lvq_prototypes[y']          (déjà calculés)
  3. recherche linéaire pour chaque prototype x_c :
       L(λ) = (1−λ)² + β·(1 − P(y'|λ·x0 + (1−λ)·x_c))²
     (1−λ)² : proximité à x0 (λ=1 ↔ x0, λ=0 ↔ prototype)
     (1−P)² : validité, P fournie par le classifieur de la classe cible
  4. validité : contrôlée par ce même classifieur f_y'
     (binaire : clf.predict ; multiclasse : clf[y'].predict == 1)
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

        # Phase OFFLINE : LVQ1 sur tout Dtrain, quel que soit le
        # classifieur (binaire ou dict OVR). L'apprentissage des
        # prototypes ne dépend pas de la façon dont le classifieur
        # a été entraîné.
        if X_train is not None and y_train is not None:
            self._fit_lvq(X_train, y_train, n_prototypes)

    # ------------------------------------------------------------------
    # LVQ1 — Algorithme 1 (Dtrain entier, toutes classes)
    # ------------------------------------------------------------------

    def _fit_lvq(self, X_train, y_train, n_prototypes,
                 max_iter=2500, gtol=1e-5, eta=0.1):
        """
        LVQ1 sur tout Dtrain (toutes classes).

        W = nprot prototypes par classe, chacun étiqueté.
        Pour chaque (x, y) :
          w* = argmin_{w in W} ||x - w||²   (toutes classes confondues)
          si classe(w*) == y  →  w* ← w* + η(x − w*)   (attraction)
          sinon               →  w* ← w* − η(x − w*)   (répulsion)
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
                    W[w_idx] -= eta * (x - W[w_idx])   # répulsion
            if np.max(np.linalg.norm(W - prev, axis=1)) < gtol:
                break

        for cls in classes:
            self.lvq_prototypes[cls] = W[labels == cls]

    # ------------------------------------------------------------------
    # Interrogation du classifieur (f_y' tout au long)
    # ------------------------------------------------------------------

    def _get_target_proba(self, x, target_class):
        """
        P(y' | x).
        Binaire      : clf.predict_proba(x)[index de y']
        Multiclasse  : clf[y'].predict_proba(x)[1]  = f_y'(x)
        """
        x = np.array(x).reshape(1, -1)
        if isinstance(self.clf, dict):
            return self.clf[target_class].predict_proba(x)[0][1]
        classes = list(self.clf.classes_)
        return self.clf.predict_proba(x)[0][classes.index(target_class)]

    def _predict_class(self, x_c, target_class=None):
        """
        Binaire      : clf.predict(x_c)
        Multiclasse  : prédiction du classifieur de la classe cible
                       f_y' — le même qui guide l'optimisation.
                       Retourne y' si f_y' reconnaît x_c, None sinon.
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
    # Recherche linéaire (identique binaire et multiclasse)
    # ------------------------------------------------------------------

    def _line_search(self, x, x_c, target_class):
        """
        Minimise L(λ) = (1−λ)² + β · (1 − P(y' | λ·x + (1−λ)·x_c))²
        via la méthode de Brent bornée sur [0, 1].

        λ = 1 ↔ x0 (instance), λ = 0 ↔ x_c (prototype).
        (1−λ)² pénalise l'éloignement de x0 ; (1−P)² pénalise le
        manque de confiance dans la classe cible : les deux termes
        sont antagonistes, le minimum s'établit près de la frontière,
        côté classe cible.
        """
        def objective(lam):
            # lam ∈ [0, 1] : 1 = x0, 0 = prototype
            x_interp = lam * x + (1 - lam) * x_c
            proba = self._get_target_proba(x_interp, target_class)
            return (1 - lam) ** 2 + self.beta * (1 - proba) ** 2

        best_l = opt.minimize_scalar(
            objective, bounds=(0, 1), method='bounded',
            options={'maxiter': self.num_iter}
        ).x
        return best_l * x + (1 - best_l) * x_c

    # ------------------------------------------------------------------
    # Génération des candidats depuis les prototypes de y'
    # ------------------------------------------------------------------

    def _compute_candidates(self, x, target_class):
        P_target = self.lvq_prototypes.get(target_class)
        if P_target is None or len(P_target) == 0:
            return None
        return np.array([self._line_search(x, x_c, target_class)
                         for x_c in P_target])

    # ------------------------------------------------------------------
    # Stratégies de sélection
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
    # Point d'entrée
    # ------------------------------------------------------------------

    def compute_counterfactual(self, x, target, N=1, model_override=None,
                               target_class=None, strategy='greedy', **kwargs):
        """
        Les prototypes sont déjà prêts (phase offline, à l'__init__).
        Ici : recherche linéaire depuis chaque prototype de y', puis
        sélection des N meilleurs candidats valides.
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
