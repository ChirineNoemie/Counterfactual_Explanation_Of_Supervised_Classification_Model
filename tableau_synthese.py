import pandas as pd
import re
import numpy as np

# 1. Load the file
df = pd.read_csv('results_iris_wine_lvq_all.csv')

# 2. Rename for scientific clarity
df = df.rename(columns={'Distance': 'Distance_L2', 'Computation_Time': 'Time_sec'})
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
# 3. Function to extract the mean from textual lists [x.x  y.y]
def parse_array_str(s):

    if pd.isna(s): return np.nan
    # If it's already a number (as Diversity sometimes is), return it as-is
    if isinstance(s, (int, float)): return s
    numbers = _NUM_RE.findall(str(s))
    return np.mean([float(n) for n in numbers]) if numbers else np.nan

# 4. Column cleanup (conversion to real numbers)
df['Distance_L2'] = df['Distance_L2'].apply(parse_array_str)
df['Distance_L1'] = df['Distance_L1'].apply(parse_array_str)
df['Density'] = df['Density'].apply(parse_array_str)
# Diversity is sometimes already a float or a list depending on the method
df['Diversity'] = df['Diversity'].apply(parse_array_str)

# 5. Compute Success Rate in %
df['Success_Rate_pct'] = (~df['Invalid']).astype(int) * 100

# 6. Metric selection (including Diversity)
metrics = ['Distance_L2', 'Distance_L1', 'Time_sec', 'Density', 'Diversity', 'Success_Rate_pct']

# 7. Grouping and statistics computation
summary = df.groupby(['Dataset', 'Method'])[metrics].agg(['mean', 'std'])

# 8. Final formatting for the report
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

# 9. Save
final_table.to_csv('results_iris_wine_lvq_all_randomforest_q1_q10.csv')
print("Table with DIVERSITY generated successfully!")
print(final_table.head(10))