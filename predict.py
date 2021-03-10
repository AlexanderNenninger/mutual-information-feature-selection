from pathlib import Path
import pickle as pkl

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit


def prepare_data(data: pd.DataFrame, target: str, columns: [(str, float)]=None, **split_kwargs):
    """Splits data into train- and a test-set and let's you iterate over n(default=5) splittings.
    For split_kwargs see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
    Returns Iterator over splittings X_train, y_train, X_test, y_test
    """
    
    if isinstance(columns, list):
        columns = [c[1] for c in columns]
    else:
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

    # Ablation Test with increasinf number of features
    # Evaluate Classifier
    evals = []
    for selected_feature in selected_features:
        # Get columns from feature matrix
        columns = np.array(features)[selected_feature.astype(np.bool)]

        for X_train, y_train, X_test, y_test in prepare_data(data, target, columns=columns):
            
            clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=1.0,
                max_depth=1,
                random_state=0
            ).fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = (y_pred == y_test).sum() / len(y_pred)
            prec = (y_pred & y_test).sum() / y_pred.sum()
            recall = (y_pred & y_test).sum() / ((y_pred & y_test).sum() + ((y_pred == 0) & (y_test == 1)).sum())
            f1 = 2 * prec * recall / (prec + recall)

            evals.append(
                {'features': columns, 'acc': acc, 'prec': prec, 'recall': recall, 'f1': f1}
            )

    for i, e in enumerate(evals):
        eval_str = '\n\t'.join(f'{k}: {v}' for k, v in e.items())
        print(
            f'Run {i}: \n\t' + eval_str
        )