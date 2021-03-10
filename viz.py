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
for idx, data in best_features.iterrows():
    fig, ax = plt.subplots(1, 2, figsize=(15, 9), sharex=True, sharey=True)
    
    fig.suptitle(idx)
    for fpr, tpr, thresholds in e.loc[data['features'], 'roc_curve']:
        
        ax[0].set_title("False positive ROC")
        ax[0].plot(thresholds, fpr)
        
        ax[1].set_title("True positive ROC")
        ax[1].plot(thresholds, tpr)

    fig.savefig(f'plots/roc_{idx}.png')
    fig.clf()