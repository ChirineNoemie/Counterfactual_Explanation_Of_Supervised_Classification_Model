import os

from sklearn.utils import shuffle
import pandas as pd
from tqdm import tqdm

from data.dataset import DatasetLoader
from utils.utils import binarize_labels, compute_diversity, standardize_data
from method.counterfactuals_v2 import CounterfactualWachter, CounterfactualMedoid, CounterfactualDiCE
from method.lvq import CounterfactualLVQ
from sklearn.svm import SVC
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from joblib import Memory

# Get the current file's directory
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# go up one level
current_file_dir = os.path.dirname(current_file_dir)

# Construct the path to the cachedir folder
cachedir = os.path.join(current_file_dir, "cachedir")

memory = Memory(cachedir, verbose=0)
DEBUG = True


# def memorized_train_model(X_train, y_train, cv_grid_size=20, random_state=42, n_folds=2):
#     model = GridSearchCV(
#         estimator=SVC(kernel='rbf', probability=True, random_state=random_state),
#         param_grid={'C': np.logspace(-1, 3, cv_grid_size), 'gamma': np.logspace(-6, 0, cv_grid_size)},
#         n_jobs=2, verbose=0, cv=n_folds  # verbose=0 pour masquer les détails
#     )
#     model.fit(X_train, y_train)
#     return model


@memory.cache
def memorized_train_model(X_train, y_train, cv_grid_size=20, random_state=42, n_folds=2):
    model = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid={
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
        },
        n_jobs=2, verbose=0, cv=n_folds
    )
    model.fit(X_train, y_train)
    return model

@memory.cache
def memorized_kde(X, step, n_folds=2):
    kde_cv = GridSearchCV(
        estimator=KernelDensity(),
        param_grid={'bandwidth': np.arange(0.1, 10.0, step)}, n_jobs=-1, cv=n_folds,
        verbose=0  # verbose=0 pour masquer les détails
    )
    kde_cv.fit(X)
    return KernelDensity(bandwidth=kde_cv.best_params_["bandwidth"]).fit(X)


# def compute_cf_and_runtime(cf_method, x_orig, y_target, N=1, **kwargs):
#     start_time = time.perf_counter()
#     xcf = cf_method.compute_counterfactual(x_orig, target=2 * y_target - 1, N=N, **kwargs)
#     end_time = time.perf_counter()
#     return xcf, end_time - start_time

def compute_cf_and_runtime(cf_method, x_orig, y_target, model, N=1, **kwargs):
    import time
    start_time = time.perf_counter()

    if isinstance(model, dict):  # OVR multiclasse
        xcf = cf_method.compute_counterfactual(
            x_orig,
            target=1,
            model_override=model[y_target],
            target_class=y_target,
            N=N,
            **kwargs
        )
    else:  # binaire
        xcf = cf_method.compute_counterfactual(
            x_orig,
            target=2 * y_target - 1,
            model_override=model,
            target_class=y_target,
            N=N,
            **kwargs
        )

    end_time = time.perf_counter()
    return xcf, end_time - start_time


@memory.cache
def memorized_run(dataset_name=None,
                  n_folds=2,
                  max_samples=10000,
                  cv_grid_size=20,
                  method='Medoid-based',
                  num_iter=100,
                  n_counterfactuals=1,
                  clustering_method='kmedoids',
                  n_clusters=-1,
                  random_state=42,
                  lambda_=10.,
                  beta=10.,
                  strategy='greedy',
                  compute_diversity=True
                  ):
    X, y = DatasetLoader(dataset_name)()
    # DEBUG: Classes information (comment out if not needed)
    # print(f"DEBUG [{dataset_name}] -> Classes originales: {np.unique(y).tolist()} | Classes cibles (y%2): {np.unique(y % 2).tolist()}")
    X, y = shuffle(X, y, random_state=random_state)
    X, y = X[:max_samples], y[:max_samples]
    y_original = y.copy()          # multiclass
    y_binary = binarize_labels(y) # binaire
    

    kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)

    results = []
    for cv_index, (train_index, test_index) in tqdm(list(enumerate(kf.split(X))),
                                                    desc=f'running kfold on {list(locals().items())[:12]}',
                                                    delay=60, leave=False):
        # X_train, y_train = X[train_index], y[train_index]
        # X_test, y_test = X[test_index], y[test_index]
        X_train, y_train = X[train_index], y_original[train_index]
        X_test, y_test = X[test_index], y_original[test_index]

        y_binary_train = y_binary[train_index]
        y_binary_test = y_binary[test_index]
        X_train, X_test = standardize_data(X_train, X_test)

        res = run_single_split(X_train=X_train, y_train=y_train,y_binary_train=y_binary_train, X_test=X_test, y_test=y_test,y_binary_test=y_binary_test,
                               cv_index=cv_index, cv_grid_size=cv_grid_size,
                               method=method, num_iter=num_iter, n_clusters=n_clusters,
                               n_counterfactuals=n_counterfactuals, clustering_method=clustering_method,
                               lambda_=lambda_, beta=beta, n_folds=n_folds, dataset_name=dataset_name,
                               max_samples=max_samples, compute_diversity=compute_diversity)
        results.extend(res)

    return pd.DataFrame(results)


def run_single_split(X_train=None, y_train=None,y_binary_train=None, X_test=None, y_test=None,y_binary_test=None, cv_index=None, cv_grid_size=20,
                     method='Medoid-based', num_iter=100, n_clusters=-1, n_counterfactuals=1, clustering_method='kmedoids',
                     lambda_=10., beta=10., n_folds=2, dataset_name=None, max_samples=3000, strategy='greedy',
                     compute_diversity=True):
    # Train classifier and fit density estimators
    #model = train_classifier(X_train, y_train, cv_grid_size=cv_grid_size, random_state=42, n_folds=n_folds)
    classes = np.unique(y_train)

    if len(classes) > 2:
        # MULTICLASSE : un classifieur binaire par classe (OVR)
        model = train_ovr_classifiers(X_train, y_train, cv_grid_size=cv_grid_size,
                                      random_state=42, n_folds=n_folds)

    else:
        # BINAIRE
        model = train_classifier(X_train, y_binary_train,
                                cv_grid_size=cv_grid_size,
                                random_state=42, n_folds=n_folds)

        # IMPORTANT: remplacer y_train/y_test par version binaire
        y_train = y_binary_train
        y_test = y_binary_test

    kernel_density_estimators = fit_density_estimators(X_train, y_train, cv_grid_size=cv_grid_size)
    cf_method = init_method(model, method=method, X_train=X_train, y_train=y_train, num_iter=num_iter,
                            n_clusters=n_clusters, clustering_method=clustering_method, lambda_=lambda_, beta=beta)

    return generate_counterfactuals(model=model, method=cf_method, kernel_density_estimators=kernel_density_estimators,
                                    X_test=X_test, y_test=y_test, cv_index=cv_index,
                                    y_test_target='opposite', n_counterfactuals=n_counterfactuals,
                                    dataset_name=dataset_name, cv_grid_size=cv_grid_size, num_iter=num_iter,
                                    max_samples=max_samples, strategy=strategy, compute_diversity=compute_diversity)


def train_classifier(X_train, y_train, cv_grid_size=20, random_state=42, n_folds=2):
    return memorized_train_model(X_train, y_train, cv_grid_size=cv_grid_size,
                                 random_state=random_state, n_folds=n_folds)

def train_ovr_classifiers(X_train, y_train, cv_grid_size=20, random_state=42, n_folds=2):
    """Entraîne un classifieur binaire par classe (stratégie One-vs-Rest).
    Compatible avec tout classifieur retourné par memorized_train_model."""
    models = {}
    for cls in np.unique(y_train):
        y_bin = (y_train == cls).astype(int)
        models[cls] = memorized_train_model(X_train, y_bin,
                                            cv_grid_size=cv_grid_size,
                                            random_state=random_state,
                                            n_folds=n_folds)
    return models

def predict_ovr(models, x):
    """Prédit la classe pour x dans un ensemble OVR.
    Utilise predict_proba pour être compatible avec tout classifieur."""
    scores = {cls: model.predict_proba(x.reshape(1, -1))[0][1]
              for cls, model in models.items()}
    return max(scores, key=scores.get)

def fit_density_estimators(X_train, y_train, cv_grid_size=20):
    kernel_density_estimators = {}
    labels = np.unique(y_train)
    for label in labels:
        X_ = X_train[y_train == label]
        kde = optimize_kde(X_, cv_grid_size=cv_grid_size)
        kernel_density_estimators[label] = kde
    return kernel_density_estimators


def optimize_kde(X, cv_grid_size=20):
    step = 1 / cv_grid_size
    return memorized_kde(X, step)


def init_method(model=None, method='Medoid-based', X_train=None, y_train=None, num_iter=100, n_clusters=-1,
                clustering_method='kmedoids', lambda_=10., beta=10.):
    if method == 'wachter':
        return CounterfactualWachter(
            clf=model,
            lambda_=lambda_, num_iter=num_iter,
        )

    if method == 'dice':
        return CounterfactualDiCE(
            clf=model,
            X_train=X_train,
            y_train=y_train, num_iter=num_iter
        )

    if method == 'Medoid-based':
        return CounterfactualMedoid-based(
            clf=model,
            beta=beta,
            X_train=X_train,
            n_clusters=n_clusters,
            num_iter=num_iter,
            cluster_method=clustering_method
        )
    
    if method == 'lvq':
        return CounterfactualLVQ(
            clf=model,
            beta=beta,
            X_train=X_train,
            y_train=y_train,
            num_iter=num_iter,
            n_prototypes=n_clusters
        )

    raise ValueError(f"Unknown method: {method}")


def generate_counterfactuals(model=None, method=None,
                             kernel_density_estimators=None, X_test=None, y_test=None,
                             cv_index=None, y_test_target='opposite', n_counterfactuals=1,
                             dataset_name=None, cv_grid_size=20, num_iter=100, max_samples=10000, strategy='greedy',
                             compute_diversity=True):
    
    if isinstance(model, dict) and method.get_class_name() == 'LVQ':
        # OVR + saisie interactive uniquement pour LVQ
        predictions = np.array([predict_ovr(model, x) for x in X_test])
        all_classes = sorted(list(model.keys()))

        print(f"\nClasses disponibles dans ce dataset : {all_classes}")
        target_class_user = None
        while target_class_user not in all_classes:
            try:
                target_class_user = int(input(
                    f"Entrez la classe cible souhaitée parmi {all_classes} : "
                ))
                if target_class_user not in all_classes:
                    print(f"Classe invalide. Choisissez parmi {all_classes}.")
            except ValueError:
                print("Entrée invalide, entrez un entier.")

        y_test_target = []
        for i, x in enumerate(X_test):
            y_pred = predictions[i]
            if y_pred == target_class_user:
                alternatives = [c for c in all_classes if c != y_pred]
                target = alternatives[0]
                print(f"  Instance {i} déjà classée dans {y_pred}, "
                    f"classe cible ajustée à {target}")
            else:
                target = target_class_user
            y_test_target.append(target)
        y_test_target = np.array(y_test_target)

    else:
        # Binaire OU méthode non-LVQ sur multiclasse
        predictions = model.predict(X_test) if not isinstance(model, dict) \
                    else np.array([predict_ovr(model, x) for x in X_test])
        if y_test_target == 'opposite':
            y_test_target = (predictions + 1) % 2

    result = []
    for i in tqdm(list(range(X_test.shape[0])), delay=120,
                  desc=f'Testing {method.get_class_name()} on {dataset_name}', leave=False):
        x_orig = X_test[i, :]
        y_orig = y_test[i]
        pred = predictions[i]
        y_target = y_test_target[i]
        correct_prediction = pred == y_orig
        if not correct_prediction:
            continue 

        kde = kernel_density_estimators[y_target]

        xcf, t = compute_cf_and_runtime(method, x_orig, y_target, model, N=n_counterfactuals, strategy=strategy)
        print("\n====================")
        print(f"x_orig: {x_orig}")
        print(f"y_orig: {y_orig}, y_target: {y_target}")
        print(f"xcf: {xcf}")
        res = format_results(model=model, cf_method=method, x_orig=x_orig, y_orig=y_orig, y_target=y_target,
                             xcf=xcf, elapsed_time=t, kde=kde, cv_index=cv_index, x_idx=i,
                             correct_prediction=correct_prediction, n_counterfactuals=n_counterfactuals,
                             dataset_name=dataset_name, cv_grid_size=cv_grid_size, num_iter=num_iter,
                             max_samples=max_samples, strategy=strategy, compute_diversity=compute_diversity)
        result.append(res)

    return result


def format_results(model=None, cf_method=None, x_orig=None, y_orig=None, y_target=None, xcf=None,
                   elapsed_time=None, kde=None, cv_index=None, x_idx=None, correct_prediction=None,
                   n_counterfactuals=1,
                   dataset_name=None, cv_grid_size=20, num_iter=100, max_samples=10000, strategy='greedy',
                   compute_diversity=True):
    if DEBUG:
        print("\n====================")
        print(f"x_orig: {x_orig}")
        print(f"y_orig: {y_orig} → y_target: {y_target}")
        print(f"xcf: {xcf}")
    # if xcf is not None:
    #     xcf = np.array(xcf)

    # if xcf is None:
    #     invalid = True
    # elif n_counterfactuals == 1:
    #     invalid = model.predict(xcf.reshape(1, -1))[0] != y_target
    # else:
    #     invalid = np.any(model.predict(xcf) != y_target)
    def safe_predict(model, x):
        if isinstance(model, dict):
            return y_target if model[y_target].predict(
                x.reshape(1, -1))[0] == 1 else -1
        return model.predict(x.reshape(1, -1))[0]

    if xcf is None:
        invalid = True

    elif n_counterfactuals == 1:
        if isinstance(model, dict):
            pred_cf = safe_predict(model, xcf)
            if DEBUG:
                print(f"[DEBUG CF] xcf = {xcf}")
                print(f"[DEBUG CF] pred_cf = {pred_cf}, target = {y_target}")
                print(f"[DEBUG CF] valid = {pred_cf == y_target}")
            invalid = pred_cf != y_target
            
        else:
            invalid = model.predict(xcf.reshape(1, -1))[0] != y_target


    else:
        if isinstance(model, dict):
            preds_cf = np.array([safe_predict(model, x) for x in xcf])
            if DEBUG:
                print(f"[DEBUG CF] xcf = {xcf}")
                print(f"[DEBUG CF] preds_cf = {preds_cf}, target = {y_target}")
                print(f"[DEBUG CF] valid = {preds_cf == y_target}")
            invalid = np.any(preds_cf != y_target)
        else:
            invalid = np.any(model.predict(xcf) != y_target)

    diversity = None
    if xcf is None:
        density = None
        computation_time = None
        distance = None
        distance_l1 = None
        counterfactual = None
    elif xcf is not None and n_counterfactuals == 1:
        density = kde.score_samples(xcf.reshape(1, -1))[0]
        computation_time = elapsed_time
        distance = np.linalg.norm(x_orig - xcf)
        distance_l1 = np.sum(np.abs(x_orig - xcf))
        counterfactual = xcf.flatten()
    else:
        density = np.array([kde.score_samples(xcf_.reshape(1, -1))[0] for xcf_ in xcf])
        computation_time = elapsed_time
        distance = np.array([np.linalg.norm(x_orig - xcf_) for xcf_ in xcf])
        distance_l1 = np.array([np.sum(np.abs(x_orig - xcf_)) for xcf_ in xcf])
        counterfactual = np.array(xcf.reshape(-1, xcf.shape[-1]))
        # Calculer la diversité seulement si demandé
        if compute_diversity:
            from utils.utils import compute_diversity as calc_diversity
            diversity = calc_diversity(counterfactual)
        else:
            diversity = None

    return {
        'Dataset': dataset_name,
        'CV_grid_size': cv_grid_size,
        'Num_iter': num_iter,
        'Max_samples': max_samples,
        'N_clusters': cf_method.n_clusters if hasattr(cf_method, 'n_clusters') else -1,
        'Test_index': x_idx,
        'Method': cf_method.get_class_name(),
        'Density': density,
        'Original_Data': x_orig,
        'Original_Data_Label': y_orig,
        'Counterfactual': counterfactual,
        'Counterfactual_Target_Label': y_target,
        'Computation_Time': computation_time,
        'Distance': distance,
        'Distance_L1': distance_l1,
        'Invalid': invalid,
        'CV_index': cv_index,
        'Clustering_type': cf_method.cluster_method if hasattr(cf_method, 'cluster_method') else 'none',
        'Correct_Prediction': correct_prediction,
        'N_counterfactuals': n_counterfactuals,
        'Lambda': cf_method.lambda_ if hasattr(cf_method, 'lambda_') else None,
        'Beta': cf_method.beta if hasattr(cf_method, 'beta') else None,
        'Diversity': diversity,
        'Strategy': strategy if hasattr(cf_method, 'strategy') else None
    }


class CounterfactualExperiment1:
    def __init__(self, dataset_name=None,
                 cv_grid_size=20,
                 n_clusters=-1,
                 num_iter=100,
                 max_samples=10000,
                 method='Medoid-based',
                 n_counterfactuals=1,
                 X_train=None,
                 y_train=None,
                 random_state=42,
                 lambda_=10.,
                 beta=10.,
                 strategy='greedy',
                 clustering_method='kmedoids',
                 compute_diversity=True,
                 n_folds=2):
        self.n_folds = n_folds
        self.strategy = strategy
        self.beta = beta
        self.lambda_ = lambda_
        self.clustering_method = clustering_method
        self.dataset_name = dataset_name
        self.cv_grid_size = cv_grid_size
        self.n_clusters = n_clusters
        self.num_iter = num_iter
        self.max_samples = max_samples
        self.method = method
        self.results = []
        self.n_counterfactuals = n_counterfactuals
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.compute_diversity = compute_diversity

    def load_data(self):
        dl = DatasetLoader(self.dataset_name)
        X, y = dl()
        return shuffle(X, y, random_state=self.random_state)

    def compute_cf_and_runtime(self, cf_method, x_orig, y_target):
        start_time = time.perf_counter()
        N = self.n_counterfactuals
        xcf = cf_method.compute_counterfactual(x_orig, target=2 * y_target - 1, N=N)
        end_time = time.perf_counter()
        return xcf, end_time - start_time

    def run(self):
        """Execute in a memorized way. This will cache the results of the experiment and avoid re-running the same exp.
        Also, I want to avoid using a sqlite database since it's a mess to keep track of the db file.
        In this way I just "imagine" I can run the experiment every time, and it will be fast and efficient.
        But in practice it is cached."""
        return memorized_run(
            dataset_name=self.dataset_name,
            n_folds=self.n_folds,
            max_samples=self.max_samples,
            cv_grid_size=self.cv_grid_size,
            method=self.method,
            num_iter=self.num_iter,
            n_counterfactuals=self.n_counterfactuals,
            clustering_method=self.clustering_method,  # self.kwargs.get('clustering_method', 'none'),
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            lambda_=self.lambda_,
            beta=self.beta,
            strategy=self.strategy
        )