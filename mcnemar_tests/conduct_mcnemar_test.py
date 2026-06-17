# This code file assesses statistical significance between the five CMTL frameworks using pairwise McNemar tests on conjoint Top-1 predictions.

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from scipy.stats import combine_pvalues

FRAMEWORKS = ['sca', 'gating', 'ffs', 'ffc', 'film']

SEEDS = [0, 1, 42, 123, 492]
BASE_PATH = "experiments/cmtl"

TASK_CONFIG = {
    'Species': {
        'file': 'species_predictions.csv',
        'calc_fn': lambda df: (df['true_species_id'] == df['pred_species_id']).astype(int)
    },
    'Disease': {
        'file': 'disease_predictions.csv',
        'calc_fn': lambda df: (df['true_disease_id'] == df['pred_disease_id']).astype(int)
    },
    'Joint': {
        'file': 'joint_predictions.csv',
        'calc_fn': lambda df: df['joint_correct'].astype(int)
    }
}

# load data
data = {task: {fw: {} for fw in FRAMEWORKS} for task in TASK_CONFIG}

for fw in FRAMEWORKS:
    for seed in SEEDS:
        seed_dir = os.path.join(BASE_PATH, fw, f"seed_{seed}")
        if not os.path.exists(seed_dir):
            seed_dir = os.path.join(BASE_PATH, fw, str(seed))

        for task, cfg in TASK_CONFIG.items():
            file_path = os.path.join(seed_dir, cfg['file'])
            if not os.path.exists(file_path):
                raise FileNotFoundError(file_path)

            df = pd.read_csv(file_path)
            df = df.sort_values('image_id').set_index('image_id')

            vec = cfg['calc_fn'](df)
            assert vec.isna().sum() == 0

            data[task][fw][seed] = vec

# McNemar Tests (PAIRWISE OVER SEEDS)
pairs = list(combinations(FRAMEWORKS, 2))
raw_results = []

for task in TASK_CONFIG:
    for fwA, fwB in pairs:

        seed_pvals = []
        seed_deltas = []

        for seed in SEEDS:

            vA = data[task][fwA][seed]
            vB = data[task][fwB][seed]

            idx = vA.index.intersection(vB.index)
            vA = vA.loc[idx]
            vB = vB.loc[idx]

            ct = pd.crosstab(vA, vB).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

            discordant = ct.iloc[0, 1] + ct.iloc[1, 0]
            exact = discordant < 25

            res = mcnemar(ct.values, exact=exact, correction=True)
            seed_pvals.append(res.pvalue)

            # effect size
            seed_deltas.append(vA.mean() - vB.mean())

        stat, combined_p = combine_pvalues(seed_pvals, method="fisher")

        raw_results.append({
            'Task': task,
            'Model_A': fwA,
            'Model_B': fwB,
            'Delta_Acc': np.mean(seed_deltas),
            'Delta_Std': np.std(seed_deltas),
            'Combined_P': combined_p,
            'Significant_Seeds': f"{sum(p < 0.05 for p in seed_pvals)}/{len(SEEDS)}"
        })

df_results = pd.DataFrame(raw_results)

# Apply Holm-Bonferroni Correction
final = []
for task, group in df_results.groupby("Task"):
    reject, p_adj, _, _ = multipletests(group["Combined_P"], method="holm")

    group = group.copy()
    group["Holm_P"] = p_adj
    group["Reject"] = reject
    final.append(group)

df_final = pd.concat(final)
df_final.to_csv("full_mcnemar_results.csv", index=False)
