"""Local airline analytics and inference application."""
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard import *
from airline import DATA, OUT, FEATURES, CATEGORICAL, NUMERIC, SERVICES, TARGET, LABELS, load_data, audit_data

st.set_page_config(page_title='Airline Passenger Satisfaction', page_icon='✈', layout='wide')
st.markdown('''<style>
.stApp {background:linear-gradient(145deg,#f3f7fc 65%,#e0f2fe)}
.block-container {padding-top:2rem;max-width:1450px}
h1,h2,h3 {color:#102b49;letter-spacing:-.035em}
[data-testid="stSidebar"] {border-right:1px solid #c9deee}
.hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#0b2039,#174a70);color:white;padding:36px 40px;border-radius:20px;margin-bottom:25px}
.hero h1 {color:white;font-size:2.65rem;max-width:850px;margin:10px 0}
.hero p {color:#c9e5f6;max-width:760px;margin:0;line-height:1.7}
.eyebrow {font-size:12px;letter-spacing:3px;font-weight:700;color:#80d1f7}
.sky {position:absolute;right:35px;top:10px;font-size:100px;opacity:.12;transform:rotate(-12deg);pointer-events:none}
.ticket {background:white;border:1px solid #d2e2ef;border-left:5px solid #34abe2;border-radius:12px;padding:20px 24px;margin-bottom:18px;box-shadow:0 6px 18px #102b4907}
.ticket .tag {color:#526d85;text-transform:uppercase;letter-spacing:1.7px;font-size:11px}
.ticket .value {font-size:32px;font-weight:750;color:#102b49;margin:5px 0}
.ticket .foot {border-top:1px dashed #bfd5e6;padding-top:10px;font-size:12px;color:#526d85}
[data-testid="stPlotlyChart"] {background:white;border-radius:12px;padding:8px}
.note {font-size:12px;color:#526d85;margin-top:24px}
</style>''', unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    return joblib.load(OUT / 'best_pipeline.joblib')

def chart(fig):
    margins = dict(l=65, r=25, t=75, b=65)
    margins.update(fig.layout.margin.to_plotly_json())
    fig.update_layout(paper_bgcolor='white', plot_bgcolor=fig.layout.plot_bgcolor or 'white', font_color='#102b49',
                      colorway=['#168ac0', '#102b49', '#76c8ef'], margin=margins,
                      title=dict(x=0, xanchor='left', font=dict(size=14)))
    fig.update_xaxes(automargin=True, title_standoff=14)
    fig.update_yaxes(automargin=True, title_standoff=14)
    st.plotly_chart(fig, use_container_width=True, theme=None)

def card(col, title, value, foot):
    col.markdown(f'<div class="ticket"><div class="tag">{title}</div><div class="value">{value}</div><div class="foot">{foot}</div></div>', unsafe_allow_html=True)

df = get_data()
st.sidebar.markdown('## ✈ AIRLINE INSIGHTS')
st.sidebar.caption('PASSENGER EXPERIENCE / ANALYTICS')
page = st.sidebar.radio('Explore', ['Overview', 'Flight Delay Analysis', 'Service Quality Analysis', 'Model comparison', 'Predict satisfaction'], label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('LOCAL RESEARCH WORKSPACE\n\nSource: supplied passenger survey CSV. No public deployment.')
st.markdown('<div class="hero"><span class="sky">☁ ✈</span><div class="eyebrow">FROM JOURNEY TO INSIGHT</div><h1>Airline Passenger Satisfaction</h1><p>Explore the passenger experience. Understand service patterns, compare predictive models, and investigate post-flight satisfaction.</p></div>', unsafe_allow_html=True)

if page == 'Overview':
    cols = st.columns(3)
    card(cols[0], 'Passengers in the dataset', f'{len(df):,}', 'BOARDING PASS · COMPLETE SAMPLE')
    card(cols[1], 'Satisfied passengers', f'{df[TARGET].mean():.2%}', 'TRUE LABEL · ASSUMED SATISFIED')
    card(cols[2], 'Median arrival delay', f'{df[NUMERIC[-1]].median():.0f} min', 'OBSERVED ARRIVAL EXPERIENCE')
    st.subheader('A clearer view of the journey')
    a,b = st.columns([3,2])
    with a:
        summary = df.groupby('Class')[TARGET].agg(passengers='size', satisfied_rate='mean').reset_index()
        fig = px.bar(summary, x='Class', y='satisfied_rate', text_auto='.2%', title='Satisfaction by cabin', hover_data=['passengers'])
        fig.update_traces(hovertemplate='%{x}<br>Satisfied: %{y:.2%}<br>Records: %{customdata[0]:d}<extra></extra>'); fig.update_yaxes(tickformat='.2%', range=[0,1]); chart(fig)
    with b:
        st.markdown('### Your flight plan')
        st.write('**01 / Flight & service dashboards** — Filter passengers and explore the Tableau-inspired analyses.')
        st.write('**02 / Model comparison** — Inspect independent test metrics and training cross-validation.')
        st.write('**03 / Predict satisfaction** — Enter a post-flight scenario using the saved pipeline.')
    st.info('Label interpretation from the supplied Tableau screenshots: True = Satisfied; False = Dissatisfied. Wi-Fi zero is labeled Not applicable in that screenshot only; other service zeros remain unspecified. The pipeline preserves all zero codes. Associations do not prove causation.')
    with st.expander('Data quality and provenance'):
        audit = audit_data(df)
        st.write(f"{len(df):,} rows · {len(df.columns)} columns · {sum(audit['missing'].values())} missing cells · {audit['duplicates']} exact duplicates")
        st.dataframe(pd.DataFrame({'dtype':df.dtypes.astype(str), 'missing':df.isna().sum()}), use_container_width=True)
        st.dataframe(pd.Series(audit['zero_ratings'], name='Zero ratings'), use_container_width=True)
        st.caption('No data dictionary, sampling dates, airline name or passenger identifier was supplied. The cleaned file may contain upstream transformations that cannot be audited here.')
        st.download_button('Download data audit', json.dumps(audit, indent=2), 'data_audit.json', 'application/json')

elif page in ['Flight Delay Analysis', 'Service Quality Analysis']:
    st.subheader(page)
    st.caption('All exploratory charts and the CSV download use these filters. Model evaluation always uses the independent saved test set.')
    a,b,c = st.columns(3)
    cabins = a.multiselect('Cabin class', CLASSES, default=CLASSES, key='cabins')
    travel = b.multiselect('Travel purpose', sorted(df['Type of Travel'].unique()), default=sorted(df['Type of Travel'].unique()), key='travel')
    customers = c.multiselect('Customer type', sorted(df['Customer Type'].unique()), default=sorted(df['Customer Type'].unique()), key='customers')
    a,b = st.columns(2)
    ages = a.slider('Age range', int(df.Age.min()), int(df.Age.max()), (int(df.Age.min()), int(df.Age.max())), key='ages')
    delay = b.slider('Arrival delay range (minutes)', 0, int(df[ARRIVAL].max()), (0, int(df[ARRIVAL].max())), key='delay')
    view = df[df['Class'].isin(cabins) & df['Type of Travel'].isin(travel) & df['Customer Type'].isin(customers) & df.Age.between(*ages) & df[ARRIVAL].between(*delay)]
    st.download_button('Download filtered passengers', view.to_csv(index=False).encode('utf-8'), 'filtered_passengers.csv', 'text/csv')
    st.caption(f'{len(view):,} matching records · Satisfaction: {pct(view[TARGET].mean())}')
    if view.empty:
        st.warning('No passengers match these filters. Widen the ranges or select more categories.')
    elif page == 'Flight Delay Analysis':
        cols = st.columns(2)
        card(cols[0], 'Average departure delay', f'{view[DEPARTURE].mean():.2f} min', 'FILTERED PASSENGERS')
        card(cols[1], 'Average arrival delay', f'{view[ARRIVAL].mean():.2f} min', 'FILTERED PASSENGERS')
        figs = flight_figures(view)
        for col,fig in zip(st.columns([3,2]),figs[:2]):
            with col: chart(fig)
        for col,fig in zip(st.columns(3),figs[2:]):
            with col: chart(fig)
        st.caption('Delayed over 30 minutes means arrival delay strictly greater than 30 minutes. Scatter points combine identical delay pairs; hover shows the number of records, with no sampling. Box plots show the median, quartiles, 1.5×IQR whiskers and outliers.')
    else:
        zeros = st.checkbox('Exclude zero codes from service averages', value=False)
        st.caption('Averages include all recorded codes by default to match the Tableau view; this is an arithmetic convention, not a claim that zero means poor service. The toggle affects boarding and dumbbell means only. Distributions and satisfaction heatmaps retain zero. Wi-Fi 0 = Not applicable per supplied screenshot; other zeros are unspecified.')
        figs = service_figures(view, exclude_zero=zeros)
        for offset in [0,2,4]:
            for col,fig in zip(st.columns(2),figs[offset:offset+2]):
                with col: chart(fig)
        st.caption('Heatmaps show the satisfied fraction within each cell, not the share of all passengers. Hover for record counts. Empty combinations are No data; an observed group with no satisfied passengers is 0.00%.')
    with st.expander('Inspect filtered records'):
        st.dataframe(view.head(500), use_container_width=True)
        st.caption('Preview limited to 500 rows. The download contains all matching records.')

else:
    if page != 'Model comparison':
        if not (OUT / 'report.json').exists() or not (OUT / 'best_pipeline.joblib').exists():
            st.error('Training artifacts are missing. Run: python airline.py')
            st.stop()
        report = json.loads((OUT / 'report.json').read_text())
        if hashlib.sha256(DATA.read_bytes()).hexdigest() != report['data_sha256']:
            st.error('Dataset differs from the training data. Retrain before using model pages.')
            st.stop()
    if page == 'Model comparison':
        from tableau_page import render
        render(chart)
    else:
        st.subheader('Your passenger boarding pass')
        st.caption(f"Post-flight prediction · {report['best_model']} · saved training-only preprocessing")
        st.info('Enter observed passenger and journey details. Wi-Fi 0 means Not applicable per the supplied screenshot; other rating-zero meanings remain unspecified. Ratings 1–5 are retained as recorded. The source does not define their verbal anchors.')
        with st.form('passenger'):
            values = {}
            st.markdown('#### Passenger & journey')
            cols = st.columns(4)
            for i,c in enumerate(CATEGORICAL):
                values[c] = cols[i].selectbox(c, sorted(df[c].unique()))
            cols = st.columns(4)
            for i,c in enumerate(NUMERIC):
                low = int(df[c].min())
                high = int(np.ceil(df[c].max()))
                values[c] = cols[i].number_input(c, min_value=low, max_value=high, value=int(df[c].median()), step=1)
            st.caption('Numeric inputs are restricted to the observed dataset ranges. Flight distance units were not supplied.')
            st.markdown('#### Service experience')
            cols = st.columns(3)
            for i,c in enumerate(SERVICES):
                values[c] = cols[i%3].select_slider(c, options=list(range(6)), value=3)
            submitted = st.form_submit_button('Predict satisfaction', type='primary')
        if submitted:
            sample = pd.DataFrame([values])[FEATURES]
            model = get_model()
            pred = int(model.predict(sample)[0])
            index = list(model.classes_).index(1)
            probability = float(model.predict_proba(sample)[0,index])
            cols = st.columns(2)
            card(cols[0], 'Predicted passenger outcome', LABELS[pred], 'POST-FLIGHT MODEL ESTIMATE')
            card(cols[1], 'Estimated satisfied probability', f'{probability:.2%}', 'UNCALIBRATED · NOT A GUARANTEE')
            st.caption('This estimate is observational and should not be used as a causal recommendation or an individual service decision.')
            result = {**values,'prediction':LABELS[pred],'satisfied_probability':probability,'model':report['best_model']}
            st.download_button('Download prediction', json.dumps(result, indent=2), 'prediction.json','application/json')

st.markdown('<div class="note">✈ AIRLINE INSIGHTS &nbsp; / &nbsp; Reproducible analysis · Observational evidence · Local workspace</div>', unsafe_allow_html=True)
