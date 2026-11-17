"""Original-project comparison only. Does not read or modify the live model's scores."""
import json
import pandas as pd
import streamlit as st
from dashboard import MODEL_ORDER, pct, accuracy_figure, confusion_figure
from tableau_results import load_verified_results, RECORDS, REPORT

def render(chart):
    st.subheader('Original project / Tableau results')
    st.info('Verified from the workbook’s embedded prediction records. These results do not evaluate the separately trained Python model used on the Predict satisfaction page.')
    try:
        report,records=load_verified_results()
    except (OSError,ValueError,KeyError) as error:
        st.error(f'Unable to verify Tableau results: {error}')
        return
    st.caption(f"Source: {report['workbook_filename']} · {report['total_records']:,} prediction records across three models. Exploratory dashboard filters do not affect this comparison.")
    left,right=st.columns([3,1])
    with left:
        fig=accuracy_figure(report)
        fig.update_layout(title='Original project / Tableau accuracy')
        chart(fig)
    with right:
        selected=st.selectbox('Select model',MODEL_ORDER)
        row=next(r for r in report['models'] if r['model']==selected)
        st.metric('Test records',f"{row['records']:,}")
    left,right=st.columns([2,3])
    with left:
        st.markdown('#### Selected-model original project metrics')
        details={'Accuracy':row['accuracy'],'Actual satisfaction rate':row['actual_satisfaction_rate'],
            'F1-score (Satisfied)':row['f1'],'Precision (Dissatisfied)':row['precision_dissatisfied'],
            'Precision (Satisfied)':row['precision'],'Predicted satisfaction rate':row['predicted_satisfaction_rate'],
            'Recall (Dissatisfied)':row['recall_dissatisfied'],'Recall (Satisfied)':row['recall']}
        st.dataframe(pd.Series({k:pct(v) for k,v in details.items()},name='Original project / Tableau').to_frame(),use_container_width=True)
        st.caption('Precision, recall and F1 use Satisfied as the positive class unless Dissatisfied is explicitly named. Confusion-matrix cells are record counts.')
    with right:
        fig=confusion_figure(row['confusion_matrix'],selected)
        fig.update_layout(title=f'{selected} · original project confusion matrix')
        chart(fig)
    st.markdown('#### All models — original project / Tableau results')
    table=pd.DataFrame(report['models']).set_index('model').reindex(MODEL_ORDER)
    st.dataframe(table[['accuracy','precision','recall','f1']].rename(columns={
        'accuracy':'Accuracy','precision':'Precision (Satisfied)','recall':'Recall (Satisfied)',
        'f1':'F1 (Satisfied)'}).style.format('{:.2%}'),use_container_width=True)
    st.caption('All rates are calculated from actual/predicted labels and stored on the 0–1 scale. Downloads retain raw numeric rates. No Python training cross-validation scores are included on this page.')
    export=table.drop(columns='confusion_matrix').reset_index().to_csv(index=False)
    st.download_button('Download original project metrics',export,'tableau_model_comparison.csv','text/csv')
    st.download_button('Download Tableau verification report',REPORT.read_bytes(),'tableau_report.json','application/json')
    st.download_button('Download Tableau prediction records',RECORDS.read_bytes(),'tableau_predictions.csv','text/csv')
    with st.expander('Verification and model provenance'):
        st.write('The Model Results datasource unions KNN_Predictions.csv, DecisionTree_Predictions.csv and NaiveBayes_Predictions.csv. The workbook filters Model KPIs and Confusion Matrix by Table Name; its accuracy comparison includes all three. Every model has actual and predicted Boolean labels with no missing labels.')
        st.write('The original exports differ from the website’s independently trained estimators. Equal test counts do not prove identical test rows. The workbook contains predictions and model-specific transformed feature columns, but no fitted estimators, preprocessing pipeline, split row IDs or training workflow. It is insufficient to reproduce original live predictions or establish the exact cause of score differences.')
        st.write('To align live predictions, provide the original serialized KNN, Decision Tree and Naive Bayes models (including the software/version needed to load them), their fitted preprocessing and feature-selection artifacts with feature order, and the training workflow/notebook plus split seed and row assignments. Original filenames are not recorded in the workbook, so exact model-file names cannot be inferred.')
        st.caption(f"Workbook SHA-256: {report['workbook_sha256']}")
