"""Validated data contract and reproducible, leakage-safe model training."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'Airline_Passenger_Cleaned.csv'
OUT = ROOT / 'artifacts'
TARGET = 'satisfaction'
SEED = 42
CATEGORICAL = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
NUMERIC = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
SERVICES = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
            'Inflight entertainment', 'On-board service', 'Leg room service',
            'Baggage handling', 'Checkin service', 'Cleanliness']
FEATURES = CATEGORICAL + NUMERIC + SERVICES
LABELS = {0: 'Dissatisfied', 1: 'Satisfied'}


def load_data(path=DATA):
    df = pd.read_csv(path)
    expected = set(FEATURES + [TARGET])
    if set(df.columns) != expected:
        raise ValueError(f'Column mismatch: missing={expected-set(df.columns)}, extra={set(df.columns)-expected}')
    mapping = {'true': 1, 'false': 0, '1': 1, '0': 0}
    y = df[TARGET].astype(str).str.strip().str.lower().map(mapping)
    if y.isna().any() or y.nunique() != 2:
        raise ValueError('Target must contain both Boolean labels (True/False or 1/0), without missing labels.')
    df[TARGET] = y.astype(int)
    for c in NUMERIC + SERVICES:
        df[c] = pd.to_numeric(df[c], errors='raise')
        if np.isinf(df[c].dropna()).any():
            raise ValueError(f'Infinite values in {c}')
        if (df[c].dropna() < 0).any():
            raise ValueError(f'Negative values in {c}')
    for c in SERVICES:
        if not df[c].dropna().isin(range(6)).all():
            raise ValueError(f'{c}: ratings must be integers 0–5')
    return df


def audit_data(df):
    return {'rows': len(df), 'columns': len(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing': df.isna().sum().astype(int).to_dict(),
            'duplicates': int(df.duplicated().sum()),
            'target_counts': {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
            'zero_ratings': (df[SERVICES] == 0).sum().astype(int).to_dict(),
            'numeric_ranges': {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC},
            'categories': {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}}


def split_data(df):
    # Exact duplicates would otherwise be able to cross the holdout boundary.
    clean = df.drop_duplicates().reset_index(drop=True)
    return train_test_split(clean[FEATURES], clean[TARGET], test_size=0.2,
                            stratify=clean[TARGET], random_state=SEED)


def make_pipeline(model):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    # Treat rating 0 as a distinct code of unknown meaning, never as worse than 1.
    ratings = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                        ('encode', OneHotEncoder(categories=[list(range(6))]*len(SERVICES),
                                                 handle_unknown='ignore', sparse_output=False))])
    prep = ColumnTransformer([('numeric', numeric, NUMERIC), ('categories', categorical, CATEGORICAL),
                              ('ratings', ratings, SERVICES)])
    return Pipeline([('preprocess', prep), ('model', model)])


def candidates():
    # Fixed, declared configurations: no test-set tuning.
    return {'Decision Tree': DecisionTreeClassifier(max_depth=18, min_samples_leaf=5, random_state=SEED),
            'KNN': KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='brute', n_jobs=2),
            'Naive Bayes': GaussianNB(var_smoothing=1e-8)}


def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)), 'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)), 'f1': float(f1_score(y, pred, zero_division=0))}


def train(path=DATA, output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = load_data(path)
    X_train, X_test, y_train, y_test = split_data(df)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    with threadpool_limits(limits=2):
        for name, model in candidates().items():
            start = time.perf_counter()
            pipeline = make_pipeline(model)
            print(f'Cross-validating {name} on {len(X_train):,} training rows...', flush=True)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=1)
            pipeline.fit(X_train, y_train)
            fitted[name] = pipeline
            rows.append({'model': name, 'cv_f1_mean': float(scores.mean()),
                         'cv_f1_std': float(scores.std()), 'cv_f1_folds': scores.tolist(),
                         'training_seconds': time.perf_counter()-start})
        best = max(rows, key=lambda r: r['cv_f1_mean'])['model']
        # Selection is locked BEFORE evaluating any model on the held-out test set.
        print(f'CV-selected model: {best}. Evaluating independent test set.', flush=True)
        predictions = pd.DataFrame({'row_id': X_test.index, 'actual': y_test.to_numpy()})
        for row in rows:
            pred = fitted[row['model']].predict(X_test)
            row.update(metrics(y_test, pred))
            row['confusion_matrix'] = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
            predictions[row['model']] = pred
    report = {'best_model': best, 'selection_metric': '3-fold training CV F1 (satisfied=1)',
              'seed': SEED, 'train_rows': len(X_train), 'test_rows': len(X_test),
              'train_prevalence': float(y_train.mean()), 'test_prevalence': float(y_test.mean()),
              'majority_baseline_accuracy': float((y_test == y_train.mode()[0]).mean()),
              'data_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'python': platform.python_version(), 'sklearn': sklearn.__version__, 'models': rows}
    joblib.dump(fitted[best], output / 'best_pipeline.joblib')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'audit.json').write_text(json.dumps(audit_data(df), indent=2), encoding='utf-8')
    pd.DataFrame(rows).drop(columns=['confusion_matrix', 'cv_f1_folds']).to_csv(output / 'model_comparison.csv', index=False)
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    np.savez_compressed(output / 'split_indices.npz', train=X_train.index.to_numpy(), test=X_test.index.to_numpy())
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    train(args.data, args.output)
