import time
import numpy as np
import pandas as pd
import torch
from raiutils.exceptions import UserConfigValidationException
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn_extra.cluster import KMedoids
import scipy.optimize as opt

from method.model import RBFKernel, RBFKernelNumpy, DifferentiableRBFSVMModel, RBFSVMModel
from utils.utils import compute_diversity

Methods = {
    'wachter': 'Wachter et al.',
    'proposal': 'Proposal',
    'dice': 'DiCE',
}


class BaseCounterfactual:
    def __init__(self, clf, num_iter=100):
        self.start_init_time = time.perf_counter()
        self.clf = clf
        self.num_iter = num_iter
        self.generation_times = []
        self._set_model(clf) 
        # if isinstance(clf, GridSearchCV):
        #     xis = clf.best_estimator_.support_vectors_
        #     alphas = abs(clf.best_estimator_.dual_coef_[0])
        #     yis = (clf.best_estimator_.dual_coef_[0] > 0) * 2 - 1
        #     b = clf.best_estimator_.intercept_[0]
        #     gamma = clf.best_estimator_.gamma
        # elif isinstance(clf, SVC):
        #     xis = clf.support_vectors_
        #     alphas = abs(clf.dual_coef_[0])
        #     yis = (clf.dual_coef_[0] > 0) * 2 - 1
        #     b = clf.intercept_[0]
        #     gamma = clf.gamma
        # else:
        #     raise ValueError('Unknown classifier type')

        # self.xis = torch.FloatTensor(xis)
        # self.alphas = torch.FloatTensor(alphas)
        # self.yis = torch.FloatTensor(yis)
        # self.b = b
        # self.gamma = gamma
        # self.kernel = RBFKernel(gamma)

    def __post_init__(self):
        self.end_init_time = time.perf_counter()
        self.init_time = self.end_init_time - self.start_init_time
    def _set_model(self, clf):
        self.clf_obj = clf  # classifieur brut conservé pour les appels génériques (predict/predict_proba)
        if isinstance(clf, dict):
            self.is_svm = True
            self.model_dict = clf
             # prendre un modèle de référence
            clf0 = next(iter(clf.values()))

            # récupérer le vrai SVM
            if isinstance(clf0, GridSearchCV):
                clf0 = clf0.best_estimator_

            xis = clf0.support_vectors_
            alphas = abs(clf0.dual_coef_[0])
            yis = (clf0.dual_coef_[0] > 0) * 2 - 1
            b = clf0.intercept_[0]
            gamma = clf0.gamma

            self.xis = np.array(xis)
            self.alphas = np.array(alphas)
            self.yis = np.array(yis)
            self.b = b
            self.gamma = gamma
            self.kernel = RBFKernel(gamma)
            return
        elif isinstance(clf, GridSearchCV):
                self.is_svm = True
                xis = clf.best_estimator_.support_vectors_
                alphas = abs(clf.best_estimator_.dual_coef_[0])
                yis = (clf.best_estimator_.dual_coef_[0] > 0) * 2 - 1
                b = clf.best_estimator_.intercept_[0]
                gamma = clf.best_estimator_.gamma
        elif isinstance(clf, SVC):
                self.is_svm = True
                xis = clf.support_vectors_
                alphas = abs(clf.dual_coef_[0])
                yis = (clf.dual_coef_[0] > 0) * 2 - 1
                b = clf.intercept_[0]
                gamma = clf.gamma
        else:
                # Classifieur générique (non-SVM) : on stocke uniquement clf_obj
                self.is_svm = False
                self.xis = np.zeros((1, 1))
                self.alphas = np.zeros(1)
                self.yis = np.zeros(1)
                self.b = 0.0
                self.gamma = 1.0
                self.kernel = RBFKernel(1.0)
                return

        # self.xis = torch.FloatTensor(xis)
        # self.alphas = torch.FloatTensor(alphas)
        # self.yis = torch.FloatTensor(yis)
        self.xis = np.array(xis)
        self.alphas = np.array(alphas)
        self.yis = np.array(yis)
        self.b = b
        self.gamma = gamma
        self.kernel = RBFKernel(gamma)
        
    @classmethod
    def get_class_name(cls):
        raise NotImplementedError

    # def decision_function(self, x):
    #     return torch.sum(self.alphas * self.yis * self.kernel(self.xis, x).flatten()) + self.b
    def decision_function(self, x):
        alphas = torch.as_tensor(self.alphas, dtype=torch.float32)
        yis = torch.as_tensor(self.yis, dtype=torch.float32)
        xis = torch.as_tensor(self.xis, dtype=torch.float32)
        b = torch.as_tensor(self.b, dtype=torch.float32)
        return torch.sum(alphas * yis * self.kernel(xis, x).flatten()) + b


    def log_generation_time(self, t):
        self.generation_times.append(t)

    def get_avg_generation_time(self):
        return np.mean(self.generation_times)

    def get_std_generation_time(self):
        return np.std(self.generation_times)

    def compute_counterfactual(self, query_instance, target, N=1, model_override=None, target_class=None, **kwargs):
        start_time = time.time()
        if model_override is not None:
            self._set_model(model_override)

        # if isinstance(query_instance, np.ndarray):
        #     query_instance = torch.tensor(query_instance, dtype=torch.float32)

        # En mode OVR multiclasse, `target` vaut toujours +1 (convention du SVM
        # binaire local). On conserve la vraie classe cible dans self.target_class.
        # En binaire (target_class non fourni), on retombe sur target (-1/+1).
        self.target_class = target_class if target_class is not None else target

        cf_instance = self._compute_counterfactual(query_instance, target, N=N, **kwargs)
        end_time = time.time()
        self.log_generation_time(end_time - start_time)

        return cf_instance

    def _compute_counterfactual(self, query_instance, target, N=1, **kwargs):
        raise NotImplementedError


class CounterfactualWachter(BaseCounterfactual):
    def __init__(self, clf, num_iter=100,
                 lambda_=10.0):
        super().__init__(clf, num_iter=num_iter)
        self.lambda_ = lambda_
        self.max_iter = num_iter

    @classmethod
    def get_class_name(cls):
        return Methods['wachter']

    def compute_loss(self, cf_instance, query_instance, target):
        fx = self.decision_function(cf_instance)
        loss1 = (fx - target) ** 2
        loss2 = torch.sum(torch.abs(cf_instance - query_instance))
        return self.lambda_ * loss1 + loss2

    def _compute_counterfactual(self, query_instance, target=1.0, N=1, _lambda=10, optimizer="adam", lr=0.01,
                                return_path=False, **kwargs):
        if N == 1:
            return self._single_counterfactual(query_instance, target, _lambda, optimizer, lr, return_path)

        results = []
        for i in range(N):
            results.append(self._single_counterfactual(query_instance, target, _lambda, optimizer, lr, return_path))

        return np.array(results).reshape(N, -1)

    def _single_counterfactual(self, query_instance, target=1.0, _lambda=10, optimizer="adam", lr=0.01,
                               return_path=False):
        self.lambda_ = _lambda

        query_instance = torch.FloatTensor(query_instance.reshape(1, -1))
        cf_instance = torch.randn(query_instance.shape) if not return_path else torch.FloatTensor(query_instance)

        optim = torch.optim.Adam([cf_instance], lr) if optimizer == "adam" else torch.optim.RMSprop([cf_instance], lr)

        path = []
        for i in range(self.max_iter):
            path.append(cf_instance.clone().detach().numpy())
            cf_instance.requires_grad = True
            optim.zero_grad()
            loss = self.compute_loss(cf_instance, query_instance, target)
            loss.backward()
            cf_instance.grad = cf_instance.grad
            optim.step()
            cf_instance.detach()

        if return_path:
            path.append(cf_instance.clone().detach().numpy())
            return cf_instance.detach().numpy(), path

        return cf_instance.detach().numpy()


def clustering_method(method, n_clusters):
    if method == 'kmedoids':
        if n_clusters == 1:
            return KMedoids(n_clusters=n_clusters, method='pam', random_state=42, max_iter=0)
        return KMedoids(n_clusters=n_clusters, method='pam', random_state=42, max_iter=1000)
    elif method == 'kmeans':
        return KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


class CounterfactualProposal(BaseCounterfactual):
    def __init__(self, clf, num_iter=100,
                 beta=10.0, X_train=None, n_clusters=-1, cluster_method='kmedoids', eps=1e-3):
        super().__init__(clf, num_iter=num_iter)
        if cluster_method is not None and n_clusters < 0:
            raise ValueError("Invalid number of clusters; "
                             "if you want to have < 0 clusters aka other modes, set cluster_method to None")
        #self.C = C = clf.best_estimator_.C
        if isinstance(clf, dict):
            clf0 = next(iter(clf.values()))  # prendre un modèle
        else:
            clf0 = clf

        self.C = C = clf0.best_estimator_.C
        self.eps = eps
        self.cluster_method = cluster_method
        self.beta = beta

        self._samples_outside_margin = None

        # self.xis = self.xis.numpy()
        # self.alphas = self.alphas.numpy()
        # self.yis = self.yis.numpy()
        self.num_iter = num_iter
        self.kernel = RBFKernelNumpy(self.gamma)

        if X_train is None:
            self.feasible_targets = np.array([])
        else:
            self.feasible_targets = {}
            fx = self.decision_function(X_train)
            for y_target in [-1, 1]:  # todo correct?
                self.feasible_targets[y_target] = self._get_all_samples_outside_margin_hyperplane(X_train, y_target,
                                                                                                  fx=fx)

        self.existing_counterfactuals = self.xis[self.alphas < C]
        self.n_clusters = n_clusters
        if n_clusters > 0:
            self.cluster = {}
            for y_target in [-1, 1]:  # todo correct?
                data = np.vstack([self.feasible_targets[y_target], self.existing_counterfactuals])
                if len(data) < n_clusters:
                    n_clusters = len(data)

                self.cluster[y_target] = clustering_method(cluster_method, n_clusters)
                self.cluster[y_target].fit(data)
        elif n_clusters == -1:
            self.cluster = None
            self._samples_outside_margin = {}
            for y_target in [-1, 1]:
                self._samples_outside_margin[y_target] = self._get_all_samples_outside_margin_hyperplane(self.xis,
                                                                                                         y_target)

        elif n_clusters == -2:
            # this is more sensible
            self.cluster = None
            self._samples_outside_margin = {}
            for y_target in [-1, 1]:
                data = np.vstack([self.feasible_targets[y_target], self.existing_counterfactuals])
                self._samples_outside_margin[y_target] = data

    @classmethod
    def get_class_name(cls):
        return Methods['proposal']

    def long_decision_function(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.decision_function(x))
        return np.array(y_pred).flatten()

    def decision_function(self, x):
        if len(x.shape) > 1 and x.shape[0] > 10:
            return self.long_decision_function(x)
        return np.sum(self.alphas[:, None] * self.yis[:, None] * np.exp(
            -self.gamma * np.sum((x - self.xis[:, None]) ** 2, axis=2)), axis=0) + self.b

    def _get_all_samples_outside_margin_hyperplane(self, X, y_target, fx=None):
        if fx is None:
            fx = self.decision_function(X)
        return X[y_target * fx >= 1 - self.eps]

    def _compute_medoid(self, X):
        dist_matrix = np.sum((X[:, None] - X[None]) ** 2, axis=2)
        idx = np.argmin(np.sum(dist_matrix, axis=1))
        return X[idx]

    def _line_search(self, x, x_c, y):
        def objective(l):
            return l ** 2 + self.beta * (self.decision_function(l * x + (1 - l) * x_c) - y) ** 2

        best_l = opt.minimize_scalar(objective, bounds=(0, 1), options={'maxiter': self.num_iter}).x
        return best_l * x + (1 - best_l) * x_c

    def _compute_starting_counterfactual_x_c(self, y):
        if self.cluster is not None:
            x_c = self.cluster[y].cluster_centers_
        elif self.n_clusters <= 0:
            x_c = self._samples_outside_margin[y]
        else:
            raise ValueError(f"Unknown clustering method: {self.cluster_method}")

        return x_c

    def _compute_counterfactual_candidates(self, x, y):

        print(f"[DEBUG] target = {y}")
        y_train_dbg = getattr(self, 'y_train', None)
        if y_train_dbg is not None:
            print(f"[DEBUG] nb points classe cible = {np.sum(y_train_dbg == y)}")
        x_c = self._compute_starting_counterfactual_x_c(y)
        print(f"[DEBUG] nb candidats AVANT filtrage = {len(x_c) if x_c is not None else 'None'}")
        if x_c is None or len(x_c) == 0:
            print("[DEBUG] Aucun candidat trouvé → retour None")
            return None

        x_lambda_c = np.array([self._line_search(x, xci, y) for xci in x_c])
        
        if len(self.existing_counterfactuals) > 0 and len(x_lambda_c) > 0:
            x_lambda_c = np.vstack([x_lambda_c, self.existing_counterfactuals])
        elif len(self.existing_counterfactuals) > 0:
            x_lambda_c = self.existing_counterfactuals
        print(f"[DEBUG] nb candidats APRÈS filtrage = {len(x_lambda_c) if x_lambda_c is not None else 'None'}")
        return x_lambda_c

    def _compute_counterfactual(self, query_instance, target, N=1, strategy='greedy', **kwargs):
        if N == 1:
            return self._compute_single_counterfactual(query_instance, target, **kwargs)

        if strategy == 'greedy':
            results = self._greedy_selection(query_instance, target, N)
        elif strategy == 'random':
            results = self._random_selection(query_instance, target, N)
        elif strategy == 'optimal':
            results = self._optimal_selection(query_instance, target, N)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if len(results) > 0:
            dim = results[0].shape[0]
            return np.array(results).reshape(-1, dim)

        return None

    def _greedy_selection(self, query_instance, target, N):
        raw_candidates = self._compute_counterfactual_candidates(query_instance, target)
        if raw_candidates is None:
            # CORRECTIF: aucun candidat (ex: pas de prototype pour cette classe) ->
            # on traite l'echantillon comme un echec (xcf=None plus haut), au lieu
            # de planter tout le batch sur list(None).
            return []
        candidates = list(raw_candidates)
        candidates_with_distance = sorted([(c, np.linalg.norm(c - query_instance)) for c in candidates],
                                          key=lambda x: x[1])

        results = []

        while candidates_with_distance:
            x_c, dst = candidates_with_distance.pop(0)
            if target * self.decision_function(x_c) > 1:
                results.append(x_c)
                if len(results) == N:
                    break

        return results

    def _random_selection(self, query_instance, target, N):
        results = []
        raw_candidates = self._compute_counterfactual_candidates(query_instance, target)
        if raw_candidates is None:
            return results
        candidates = list(raw_candidates)
        # shuffle candidates (dont sort, create a lazy iterator to save computational time
        # by not computing the distances for everything)
        candidates = np.random.permutation(candidates)

        def lazy_distance(candidates):
            for c in candidates:
                yield c, np.linalg.norm(c - query_instance)

        candidates_with_distance = lazy_distance(candidates)

        while candidates_with_distance:
            x_c = next(candidates_with_distance)[0]
            if target * self.decision_function(x_c) > 1:
                results.append(x_c)
                if len(results) == N:
                    break

        return results

    def _optimal_selection(self, query_instance, target, N):
        # check exhaustively the best combination of counterfactuals according to diversity()
        raw_candidates = self._compute_counterfactual_candidates(query_instance, target)
        if raw_candidates is None:
            return []
        candidates = list(raw_candidates)
        candidates_with_distance = [(c, np.linalg.norm(c - query_instance)) for c in candidates]

        # use itertools to generate all possible combinations of counterfactuals
        from itertools import combinations
        best_diversity = 0
        best_comb = None
        for comb in combinations(candidates_with_distance, N):
            if compute_diversity([c[0] for c in comb]) > best_diversity:
                best_diversity = compute_diversity([c[0] for c in comb])
                best_comb = comb

        return [c[0] for c in best_comb]

    def _compute_single_counterfactual(self, query_instance, target, **kwargs):
        raw_candidates = self._compute_counterfactual_candidates(query_instance, target)
        if raw_candidates is None:
            return None
        candidates = sorted(list(raw_candidates),
                            key=lambda p: np.linalg.norm(p - query_instance))

        while candidates:
            x_c = candidates.pop(0)
            if target * self.decision_function(x_c) > 1:
                return x_c

        # failed
        return None


class CounterfactualDiCE(BaseCounterfactual):
    def __init__(self, clf, num_iter=100,
                 backend='PYT', method='gradient', X_train=None, y_train=None):
        super().__init__(clf, num_iter=num_iter)
        self.backend = backend

        if backend == 'PYT':
            self.model = DifferentiableRBFSVMModel(self.alphas, self.xis, self.yis, self.b, self.gamma)
        else:
            self.model = RBFSVMModel(self.alphas, self.xis, self.yis, self.b, self.gamma)

        from dice_ml import Dice, Model, Data
        import pandas as pd

        dim = self.xis.shape[1]
        self.method = method
        self.num_iter = num_iter

        if X_train is None or y_train is None:
            return

        df = pd.DataFrame(X_train)
        df['label'] = y_train

        # add some very small noise to features, otherwise DiCE will not work (except to the label)
        for i in range(dim):
            df[i] += np.random.normal(0, 0.001, df.shape[0])
            # the noise must be not too small otherwise there will be another error...

        continuous_features_precision = {}
        for i in range(dim):
            continuous_features_precision[i] = 10  # should be ok

        self.dice_data = Data(dataframe=df, continuous_features=list(range(dim)),
                              outcome_name='label', continuous_features_precision=continuous_features_precision)
        self.dice_model = Model(model=self.model, backend=backend)
        self.dice_instance = Dice(self.dice_data, self.dice_model, method=method)

    @classmethod
    def get_class_name(cls):
        return Methods['dice']

    def _compute_counterfactual(self, query_instance, target, N=1, **kwargs):
        if len(query_instance.shape) != 2:
            query_instance = query_instance.reshape(1, -1)
        df_query = pd.DataFrame(query_instance)
        df_query.columns = range(query_instance.shape[1])

        import sys
        import os

        try:
            # Suppress output
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

            res = self.dice_instance.generate_counterfactuals(df_query, total_CFs=N,
                                                              desired_class='opposite',
                                                              min_iter=1,
                                                              max_iter=self.num_iter,
                                                              limit_steps_ls=self.num_iter)

            result = res.cf_examples_list[0].final_cfs_df.drop('label', axis=1).values
        except UserConfigValidationException:
            return None
        finally:
            # Restore output
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        if len(list(result)) != N:
            return None  # invalid result (not enough counterfactuals)
        if N == 1:
            return result[0]
        dim = result.shape[1]
        return np.array(list(result)).reshape(N, dim)


# what i tried doing was
# if len(xcf) < n_counterfactuals:
#             # i need to pad the outputs of dice
#             xcf = np.concatenate([xcf, np.zeros((n_counterfactuals - len(xcf), xcf.shape[-1]))])
# but this leads to headaches for the evaluation later. so i'll be stricter.

# def show_toy(xcf1_explainer, xcf2_explainer, X_train, y_train, x_orig, target):
#     res1 = xcf1_explainer.compute_counterfactual(x_orig, target=target, return_path=True)
#     xcf1, path = res1[0].flatten(), res1[1]
#     res2 = xcf2_explainer.compute_counterfactual(x_orig, target=target)
#     xcf2 = res2.flatten()
#     xis = xcf1_explainer.xis

#     import matplotlib.pyplot as plt
#     # change dpi
#     plt.figure(dpi=300)
#     # change scale font
#     plt.rcParams.update({'font.size': 16})
#     plt.xlim(-2, 4)
#     plt.ylim(-2, 2.5)
#     plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='r', label='-1', s=50)
#     plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='b', label='1', s=50)
#     plt.scatter(x_orig[0], x_orig[1], c='g', marker='x', label='Original', s=200)
#     plt.scatter(xcf1[0], xcf1[1], c='orange', marker='x', label='Baseline', s=200)
#     plt.scatter(xcf2[0], xcf2[1], c='m', marker='x', label='Proposal', zorder=30, s=200)

#     # sv
#     plt.scatter(xis[:, 0], xis[:, 1], c='k',
#                 marker='x', label='SV')
#     xx, yy = np.meshgrid(np.linspace(-2, 4, 100), np.linspace(-2, 2.5, 100))
#     plt.contourf(xx, yy, xcf1_explainer.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape),
#                  alpha=0.2,
#                  levels=20)  # linewidths=1)
#     plt.colorbar()
#     plt.contour(xx, yy, xcf1_explainer.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape),
#                 levels=[-1, 0, 1], linewidths=3,
#                 colors='black', linestyles=['--', '-', '--'])

#     path = np.array(path).reshape(-1, 2)
#     plt.plot(path[:, 0], path[:, 1], color='orange')
#     plt.plot([x_orig[0], xcf2[0]], [x_orig[1], xcf2[1]], color='m')
#     plt.legend(loc='lower left', framealpha=1.0)

#     # plt.show()
#     plt.savefig(f"toy.pdf")




# import numpy as np
# import matplotlib.pyplot as plt

# def show_toy(xcf1_explainer, xcf2_explainer, X_train, y_train, x_orig, target=1, N=1, strategy='greedy'):
#     # 1. Calcul des contre-factuels
#     # Baseline (Wachter) - toujours N=1 car il ne supporte pas les multiples
#     res1 = xcf1_explainer.compute_counterfactual(x_orig, target=target, return_path=True)
#     xcf1, path = res1[0], res1[1]
    
#     # Proposal - peut supporter N multiples
#     if N == 1:
#         xcf2 = xcf2_explainer.compute_counterfactual(x_orig, target=target)
#     else:
#         xcf2 = xcf2_explainer.compute_counterfactual(x_orig, target=target, N=N, strategy=strategy)
    
#     # 2. Correction des dimensions
#     xcf1 = np.array(xcf1).flatten()
#     xis = xcf1_explainer.xis

#     # 3. Configuration graphique
#     plt.figure(figsize=(10, 8), dpi=300)
#     plt.rcParams.update({'font.size': 16})
#     plt.xlim(-2, 4)
#     plt.ylim(-2, 2.5)

#     # 4. Dessin de la fonction de décision (Contours)
#     xx, yy = np.meshgrid(np.linspace(-2, 4, 100), np.linspace(-2, 2.5, 100))
#     grid_points = np.c_[xx.ravel(), yy.ravel()]
    
#     Z = xcf2_explainer.decision_function(grid_points).reshape(xx.shape)

#     plt.contourf(xx, yy, Z, alpha=0.2, levels=20, cmap='RdBu')
#     plt.colorbar(label="Score de décision")
#     plt.contour(xx, yy, Z, levels=[-1, 0, 1], linewidths=3,
#                 colors='black', linestyles=['--', '-', '--'])

#     # 5. Affichage des points de données
#     plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='r', label='Classe -1', s=50, alpha=0.5)
#     plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='b', label='Classe 1', s=50, alpha=0.5)
    
#     # Vecteurs Supports
#     plt.scatter(xis[:, 0], xis[:, 1], c='k', marker='x', label='SV', alpha=0.3)

#     # 6. Affichage des résultats contre-factuels
#     plt.scatter(x_orig[0], x_orig[1], c='g', marker='x', label='Original', s=200, zorder=35)
    
#     # Baseline (Wachter) - toujours un seul
#     if xcf1 is not None:
#         plt.scatter(xcf1[0], xcf1[1], c='orange', marker='x', label='Baseline (Wachter)', s=200, zorder=35)
#         # Dessin du chemin pour la baseline
#         path = np.array(path).reshape(-1, 2)
#         plt.plot(path[:, 0], path[:, 1], color='orange', linestyle='-', linewidth=2)

#     # Proposal - peut être multiple
#     if xcf2 is not None:
#         if N == 1:
#             # Un seul contre-factuel
#             xcf2 = np.array(xcf2).flatten()
#             plt.scatter(xcf2[0], xcf2[1], c='m', marker='x', label='Proposal (Ours)', s=200, zorder=40)
#             plt.plot([x_orig[0], xcf2[0]], [x_orig[1], xcf2[1]], color='m', linestyle='--', linewidth=2)
#         else:
#             # Multiples contre-factuels
#             if len(xcf2.shape) == 1:
#                 xcf2 = xcf2.reshape(1, -1)  # Au cas où on n'a qu'un seul résultat
            
#             for i in range(len(xcf2)):
#                 label = 'Proposal (Ours)' if i == 0 else None  # Label seulement pour le premier
#                 plt.scatter(xcf2[i, 0], xcf2[i, 1], c='m', marker='x', s=200, zorder=40, label=label)
#                 plt.plot([x_orig[0], xcf2[i, 0]], [x_orig[1], xcf2[i, 1]], color='m', linestyle='--', linewidth=2, alpha=0.7)

#     plt.legend(loc='lower left', framealpha=1.0, fontsize=12)
#     title = f"Comparaison des méthodes de Contrefactuels"
#     if N > 1:
#         title += f" (N={N}, Strategy={strategy})"
#     plt.title(title)
    
#     # Génération automatique du nom de fichier avec compteur
#     import os
#     counter = 1
#     while os.path.exists(f"toy_{counter}.pdf"):
#         counter += 1
#     filename = f"toy_{counter}.pdf"
    
#     plt.savefig(filename, bbox_inches='tight')
#     print(f"Graphique sauvegardé sous '{filename}' avec N={N} contre-factuels")
    
#     # Affichage des prototypes/contre-factuels obtenus
#     print("\n" + "="*60)
#     print("PROTOTYPES/CONTRE-FACTUELS OBTENUS:")
#     print("="*60)
    
#     print(f"\nPoint original: {x_orig}")
#     print(f"Target: {target}")
    
#     if xcf1 is not None:
#         print(f"\nBaseline (Wachter):")
#         print(f"  Contre-factuel: {xcf1}")
#         print(f"  Distance à l'original: {np.linalg.norm(xcf1 - x_orig):.4f}")
#         print(f"  Score de décision: {xcf2_explainer.decision_function(xcf1.reshape(1, -1))[0]:.4f}")
    
#     if xcf2 is not None:
#         print(f"\nProposal (Ours):")
#         if N == 1:
#             print(f"  Contre-factuel: {xcf2}")
#             print(f"  Distance à l'original: {np.linalg.norm(xcf2 - x_orig):.4f}")
#             print(f"  Score de décision: {xcf2_explainer.decision_function(xcf2.reshape(1, -1))[0]:.4f}")
#         else:
#             print(f"  Nombre de contre-factuels: {len(xcf2)}")
#             for i, cf in enumerate(xcf2):
#                 print(f"  CF {i+1}: {cf}")
#                 print(f"    Distance: {np.linalg.norm(cf - x_orig):.4f}")
#                 print(f"    Score: {xcf2_explainer.decision_function(cf.reshape(1, -1))[0]:.4f}")
            
#             # Calcul de la diversité si multiple
#             if len(xcf2) > 1:
#                 from utils.utils import compute_diversity
#                 diversity = compute_diversity(xcf2)
#                 print(f"  Diversité des solutions: {diversity:.4f}")
    
#     print("="*60)

#     import sys
#     sys.exit(0)


