import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

evaluation = pd.read_pickle('output/evals.pkl')

# Create Hashable column for grouping
evaluation['features_str'] = evaluation.features.apply(str)

# Find Numeric columns for statistics
cols_numeric = evaluation.dtypes == 'float64'
cols_numeric[7] = True
cols_numeric[8] = True

# Compute group statistics
mean_scores = evaluation.loc[:, cols_numeric].groupby(['features_str', 'selection_criterion']).mean()
max_scores = evaluation.loc[:, cols_numeric].groupby(['features_str', 'selection_criterion']).max()
idx_max_scores = evaluation.loc[:, cols_numeric].groupby(['features_str', 'selection_criterion']).idxmax()

# Best Features
best_features = mean_scores.idxmax()
best_features.name = 'features'
maxes = mean_scores.max()
maxes.name = 'value'
best_features = best_features.to_frame().join(maxes)
best_features.to_csv('output/best_features.csv')

e = evaluation.set_index(['features_str', 'selection_criterion'])

# Plot ROC Curves
num_metrics = best_features.shape[0]
fig, ax = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5), sharex=True, sharey=True)
for i, (idx, data) in enumerate(best_features.iterrows()):
    for fpr, tpr, thresholds in e.loc[data['features'], 'roc_curve']:
        ax[i].set_title(f"ROC of Crossvalidation Runs with best {idx}")
        ax[i].plot(fpr, tpr)
        ax[i].set_xlabel("False Positive Rate")
        ax[i].set_ylabel("True Positive Rate")

plt.tight_layout()
fig.savefig(f'plots/roc.png')
fig.clf()