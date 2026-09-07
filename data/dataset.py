# from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine, make_moons, fetch_openml
# import numpy as np
# import pandas as pd
# import requests
# import socket
# from urllib.error import URLError
# import time


# class DatasetLoader:
#     def __init__(self, dataset_name):
#         self.dataset_name = dataset_name
#         self.X, self.y = self.load()

#     def test_internet_connection(self, timeout=5):
#         """Test if internet connection is available"""
#         try:
#             socket.create_connection(("8.8.8.8", 53), timeout=timeout)
#             return True
#         except (socket.error, OSError):
#             return False

#     def load_with_retry(self, load_func, max_retries=3, delay=2):
#         """Load dataset with retry mechanism"""
#         for attempt in range(max_retries):
#             try:
#                 print(f"    🔄 Tentative {attempt + 1}/{max_retries} de chargement...")
#                 return load_func()
#             except Exception as e:
#                 print(f"    ⚠️  Tentative {attempt + 1} échouée: {e}")
#                 if attempt < max_retries - 1:
#                     print(f"    ⏳ Attente de {delay} secondes avant nouvelle tentative...")
#                     time.sleep(delay)
#                     delay *= 2  # Exponential backoff
#                 else:
#                     raise e

#     def load(self):
#         print(f"🔍 Chargement du dataset: {self.dataset_name}")
        
#         if self.dataset_name == 'iris':
#             return load_iris(return_X_y=True)
#         elif self.dataset_name == 'digits':
#             return load_digits(return_X_y=True)
#         elif self.dataset_name == 'breast_cancer':
#             return load_breast_cancer(return_X_y=True)
#         elif self.dataset_name == 'wine':
#             return load_wine(return_X_y=True)
#         elif self.dataset_name == 'moons':
#             return make_moons(n_samples=200, noise=0.2, random_state=44)
#         # elif self.dataset_name == 'mnist':
#         #     print("    🌐 MNIST nécessite une connexion internet (OpenML)")
#         #     if not self.test_internet_connection():
#         #         raise ConnectionError("Pas de connexion internet pour charger MNIST")
            
#         #     def load_mnist():
#         #         mnist = fetch_openml('mnist_784', version=1, parser='auto')
#         #         y = np.array([int(label) for label in mnist.target])
#         #         X = np.array(mnist.data)
#         #         return X, y
            
#         #     return self.load_with_retry(load_mnist)
            
#         elif self.dataset_name == 'boston':
#             print("    🌐 Boston nécessite une connexion internet (CMU)")
#             if not self.test_internet_connection():
#                 raise ConnectionError("Pas de connexion internet pour charger Boston")
            
#             def load_boston():
#                 data_url = "http://lib.stat.cmu.edu/datasets/boston"
#                 raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
#                 X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
#                 y = raw_df.values[1::2, 2] >= 20
#                 return X, y
            
#             return self.load_with_retry(load_boston)

#         elif self.dataset_name == 'magic':
#             print("    🌐 Magic nécessite une connexion internet (UCI)")
#             if not self.test_internet_connection():
#                 raise ConnectionError("Pas de connexion internet pour charger Magic")
            
#             def load_magic():
#                 data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
#                 raw_df = pd.read_csv(data_url, header=None)
#                 X = raw_df.values[:, :-1]
#                 y = raw_df.values[:, -1] == 'g'
#                 return X, y
            
#             return self.load_with_retry(load_magic)

#         elif self.dataset_name == 'banknote':
#             print("    🌐 Banknote nécessite une connexion internet (UCI)")
#             if not self.test_internet_connection():
#                 raise ConnectionError("Pas de connexion internet pour charger Banknote")
            
#             def load_banknote():
#                 data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
#                 raw_df = pd.read_csv(data_url, header=None)
#                 X = raw_df.values[:, :-1]
#                 y = raw_df.values[:, -1] == 1
#                 return X, y
            
#             return self.load_with_retry(load_banknote)

#         else:
#             raise ValueError(f"Unknown dataset: {self.dataset_name}")

#     def __call__(self, *args, **kwargs):
#         return self.X, self.y



import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine, make_moons

DATASETS_DIR = 'datasets'

INTERNET_DATASETS = {
    'boston': "http://lib.stat.cmu.edu/datasets/boston",
    'magic': "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data",
    'banknote': "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
}

def load_from_internet(name):
    url = INTERNET_DATASETS[name]
    if name == 'boston':
        raw = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
        X = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
        y = raw.values[1::2, 2] >= 20
    else:
        raw = pd.read_csv(url, header=None)
        X = raw.values[:, :-1].astype(float)
        y = raw.values[:, -1] == ('g' if name == 'magic' else 1)
    return X, y

def load_internet_dataset(name):
    os.makedirs(DATASETS_DIR, exist_ok=True)
    path_X = f'{DATASETS_DIR}/{name}_X.npy'
    path_y = f'{DATASETS_DIR}/{name}_y.npy'

    if os.path.exists(path_X):
        return np.load(path_X), np.load(path_y)

    X, y = load_from_internet(name)
    np.save(path_X, X)
    np.save(path_y, y)
    return X, y

class DatasetLoader:
    SKLEARN_DATASETS = {
        'iris': load_iris,
        'digits': load_digits,
        'breast_cancer': load_breast_cancer,
        'wine': load_wine,
    }

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.X, self.y = self.load()

    def load(self):
        if self.dataset_name in self.SKLEARN_DATASETS:
            return self.SKLEARN_DATASETS[self.dataset_name](return_X_y=True)
        
        if self.dataset_name == 'moons':
            return make_moons(n_samples=200, noise=0.2, random_state=44)
        
        if self.dataset_name in INTERNET_DATASETS:
            return load_internet_dataset(self.dataset_name)

        raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def __call__(self):
        return self.X, self.y