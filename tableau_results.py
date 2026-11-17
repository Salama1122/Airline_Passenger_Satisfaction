"""Verify the original project's exported predictions, independently of Python training."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from airline import ROOT

SOURCE_NAMES = {'KNN_Predictions.csv': 'KNN', 'DecisionTree_Predictions.csv': 'Decision Tree',
                'NaiveBayes_Predictions.csv': 'Naive Bayes'}
RECORDS = ROOT / 'data' / 'tableau' / 'verified_predictions.csv'
REPORT = ROOT / 'artifacts' / 'tableau_report.json'

def binary_labels(values):
    result = values.astype(str).str.strip().str.lower().map({'true':1,'false':0,'1':1,'0':0})
    if result.isna().any():
        raise ValueError('Missing or unrecognized actual/predicted labels in Tableau records.')
    return result.astype(int)

def normalize_records(raw):
    required = {'Table Name','satisfaction','prediction(satisfaction)'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing Tableau prediction fields: {required-set(raw.columns)}')
    if set(raw['Table Name'].unique()) != set(SOURCE_NAMES):
        raise ValueError('Expected embedded records for all three original model exports.')
    # Export ordinal is provenance only: it is NOT a passenger ID and does not join to Python rows.
    return pd.DataFrame({'export_record':range(len(raw)), 'model':raw['Table Name'].map(SOURCE_NAMES),
                         'actual':binary_labels(raw['satisfaction']),
                         'predicted':binary_labels(raw['prediction(satisfaction)'])})

def calculate_results(records):
    if set(records['model'].unique()) != set(SOURCE_NAMES.values()):
        raise ValueError('Missing or unexpected original model records.')
    y_all, p_all = binary_labels(records['actual']), binary_labels(records['predicted'])
    results=[]
    for name in SOURCE_NAMES.values():
        mask=records['model']==name
        y,p=y_all[mask],p_all[mask]
        results.append({'model':name,'records':int(mask.sum()),
            'accuracy':float(accuracy_score(y,p)),
            'precision':float(precision_score(y,p,zero_division=0)),
            'recall':float(recall_score(y,p,zero_division=0)),
            'f1':float(f1_score(y,p,zero_division=0)),
            'precision_dissatisfied':float(precision_score(y,p,pos_label=0,zero_division=0)),
            'recall_dissatisfied':float(recall_score(y,p,pos_label=0,zero_division=0)),
            'actual_satisfaction_rate':float(y.mean()), 'predicted_satisfaction_rate':float(p.mean()),
            'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist()})
    return results

def load_verified_results():
    metadata=json.loads(REPORT.read_text(encoding='utf-8'))
    if hashlib.sha256(RECORDS.read_bytes()).hexdigest()!=metadata['records_sha256']:
        raise ValueError('Tableau records checksum mismatch. Re-import the workbook.')
    records=pd.read_csv(RECORDS)
    calculated=calculate_results(records)
    if calculated!=metadata['models']:
        raise ValueError('Saved Tableau report does not match recalculated prediction metrics.')
    return metadata,records
