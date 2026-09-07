import pandas as pd
import re
import numpy as np

# 1. Charger le fichier
df = pd.read_csv('results_iris_wine_lvq_all.csv')

# 2. Renommer pour la clarté scientifique
df = df.rename(columns={'Distance': 'Distance_L2', 'Computation_Time': 'Time_sec'})
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
# 3. Fonction pour extraire la moyenne des listes textuelles [x.x  y.y]
def parse_array_str(s):

    if pd.isna(s): return np.nan
    # Si c'est déjà un nombre (comme pour Diversity parfois), on le retourne
    if isinstance(s, (int, float)): return s
    numbers = _NUM_RE.findall(str(s))
    return np.mean([float(n) for n in numbers]) if numbers else np.nan

# 4. Nettoyage des colonnes (conversion en nombres réels)
df['Distance_L2'] = df['Distance_L2'].apply(parse_array_str)
df['Distance_L1'] = df['Distance_L1'].apply(parse_array_str)
df['Density'] = df['Density'].apply(parse_array_str)
# Diversity est parfois déjà un float ou une liste selon la méthode
df['Diversity'] = df['Diversity'].apply(parse_array_str)

# 5. Calcul du Success Rate en %
df['Success_Rate_pct'] = (~df['Invalid']).astype(int) * 100

# 6. Sélection des métriques (Inclusion de Diversity)
metrics = ['Distance_L2', 'Distance_L1', 'Time_sec', 'Density', 'Diversity', 'Success_Rate_pct']

# 7. Groupement et calcul des statistiques
summary = df.groupby(['Dataset', 'Method'])[metrics].agg(['mean', 'std'])

# 8. Formatage final pour le rapport
def format_final(row, is_pct=False):
    m = row['mean']
    s = row['std']
    if pd.isna(m): return "N/A"
    if is_pct:
        return f"{m:.1f}% ± {s:.1f}%"
    return f"{m:.3f} ± {s:.3f}"

final_table = pd.DataFrame(index=summary.index)
final_table['Distance_L2'] = summary['Distance_L2'].apply(format_final, axis=1)
final_table['Distance_L1'] = summary['Distance_L1'].apply(format_final, axis=1)
final_table['Time (sec)'] = summary['Time_sec'].apply(format_final, axis=1)
final_table['Density'] = summary['Density'].apply(format_final, axis=1)
final_table['Diversity'] = summary['Diversity'].apply(format_final, axis=1)
final_table['Success Rate'] = summary['Success_Rate_pct'].apply(lambda x: format_final(x, is_pct=True), axis=1)

# 9. Sauvegarde
final_table.to_csv('results_iris_wine_lvq_all_randomforest_q1_q10.csv')
print("Tableau avec DIVERSITÉ généré avec succès !")
print(final_table.head(10))