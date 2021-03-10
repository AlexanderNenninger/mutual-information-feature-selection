from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve


def prepare_data(data: pd.DataFrame, target: str, columns: [str]=None, **split_kwargs):
    """Splits data into train- and a test-set and let's you iterate over n(default=5) splittings.
    For split_kwargs see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    Returns Iterator over splittings X_train, y_train, X_test, y_test
    """
    
    if not isinstance(columns, list):
        columns = data.columns

    y = data.loc[:, target].to_numpy()
    X = data.loc[:, set(columns) - {target}].to_numpy()

    # Default values
    skwargs = dict(n_splits=5, test_size=0.2, random_state=0)
    # Overwrite if passed
    if isinstance(split_kwargs, dict):
        for k, v in split_kwargs.items():
            skwargs[k] = v

    sss = StratifiedShuffleSplit(**split_kwargs)

    for train_index, test_index in sss.split(X, y):
        yield X[train_index], y[train_index], X[test_index], y[test_index]

def eval_model(data: pd.DataFrame, columns: [str], target: str, classifier, **eval_info) -> [dict]:
    "Stratified crossvalidation on 'columns' of 'data' using 'classifier'"
    
    evals = []
    for X_train, y_train, X_test, y_test in prepare_data(data, target, columns=columns):
        # Fit classifier
        classifier.fit(X_train, y_train)
        y_proba = classifier.predict_proba(X_test)
        y_pred = classifier.predict(X_test)
        
        # Calculate performance metrics
        acc = (y_pred == y_test).sum() / len(y_pred)
        prec = (y_pred & y_test).sum() / y_pred.sum()
        recall = (y_pred & y_test).sum() / ((y_pred & y_test).sum() + ((y_pred == 0) & (y_test == 1)).sum())
        f1 = 2 * prec * recall / (prec + recall)
        roc_c = roc_curve(y_test, y_proba[:, 1])
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        
        # Assemble summary
        summary = {
                'features': columns,
                'acc': acc,
                'prec': prec,
                'recall': recall,
                'f1': f1,
                'roc_curve': roc_c,
                'roc_auc': roc_auc
            }

        # Add extra information to evaluation report
        for k, v in eval_info.items():
            summary[k] = v
        
        evals.append(
            summary
        )
    return evals


if __name__=='__main__':
    data_file = 'data/heart.csv'
    target = 'target'
    
    data = pd.read_csv(data_file)
    
    # Load scores
    with Path('output/scores.pkl').open('rb') as f:
        scores = pkl.load(f)
    # Load Features
    with Path('output/selected_features.pkl').open('rb') as f:
        features, selected_features = pkl.load(f)

    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=1.0,
        max_depth=1,
        random_state=0
    )

    # Ablation Test with increasing number of features - compare max MI-score vs max entropy
    # Evaluate Classifier
    evals = []
    for idx, selected_feature in enumerate(selected_features):
        # Get columns from feature matrix
        columns_mi = np.array(features)[selected_feature.astype(np.bool)]
        evaluation_mi = eval_model(data, columns_mi, 'target', clf, selection_criterion='mutual_information')
        evals.extend(evaluation_mi)

        # Get columns with max entropy
        columns_me = features[:idx+1]
        evaluation_ma = eval_model(data, columns_me, 'target', clf, selection_criterion='max_entropy')
        evals.extend(evaluation_ma)
    
    # Save evaluation
    pd.DataFrame.from_dict(evals).to_csv('output/evals.csv')

    for i, e in enumerate(evals):
        eval_str = '\n\t'.join(f'{k}: {v}' for k, v in e.items())
        print(
            f'Run {i}: \n\t' + eval_str
        )
    