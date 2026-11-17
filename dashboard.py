"""Shared dashboard aggregations/figures. Rates remain on the 0–1 scale."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from airline import TARGET, LABELS

ARRIVAL = 'Arrival Delay in Minutes'
DEPARTURE = 'Departure Delay in Minutes'
GROUP = 'Arrival delay group'
DELAY_GROUPS = ['No delay', '1–30 minutes', '31–60 minutes', '61–120 minutes', 'Over 120 minutes']
CLASSES = ['Business', 'Eco', 'Eco Plus']
MODEL_ORDER = ['KNN', 'Decision Tree', 'Naive Bayes']
ASPECTS = ['Checkin service', 'Cleanliness', 'Inflight entertainment', 'On-board service']
PURPLE = {LABELS[0]: '#e9bddd', LABELS[1]: '#794b80'}
SERVICE_COLORS = {LABELS[0]: '#b9dff1', LABELS[1]: '#ffbf80'}
YELLOW = '#edc83e'

def pct(value):
    return 'No data' if pd.isna(value) else f'{value:.2%}'

def delay_groups(values):
    # Nonnegative values: zero, (0,30], (30,60], (60,120], (120,infinity).
    return pd.cut(values, [-np.inf, 0, 30, 60, 120, np.inf], labels=DELAY_GROUPS)

def rate_summary(df, keys):
    return df.groupby(keys, observed=False)[TARGET].agg(count='size', rate='mean')

def delay_summary(df):
    return rate_summary(df.assign(**{GROUP: delay_groups(df[ARRIVAL])}), GROUP).reindex(DELAY_GROUPS)

def satisfaction_grid(df, row, column, rows, columns):
    index = pd.MultiIndex.from_product([rows, columns], names=[row, column])
    stats = rate_summary(df, [row,column]).reindex(index)
    counts = stats['count'].fillna(0).astype(int).unstack(column).reindex(index=rows, columns=columns)
    rates = stats['rate'].unstack(column).reindex(index=rows, columns=columns)
    return rates, counts

def mean_ratings(df, exclude_zero=False):
    work = df[ASPECTS + [TARGET]].copy()
    if exclude_zero:
        work[ASPECTS] = work[ASPECTS].replace(0, np.nan)
    long = work.melt(id_vars=TARGET, var_name='Service aspect', value_name='Rating')
    return long.groupby(['Service aspect', TARGET])['Rating'].agg(mean='mean', count='count').reindex(
        pd.MultiIndex.from_product([ASPECTS,[0,1]], names=['Service aspect',TARGET]))

def class_composition(df):
    counts = pd.crosstab(df['Class'],df[TARGET]).reindex(index=CLASSES, columns=[0,1],fill_value=0)
    return counts.div(counts.sum(axis=1).replace(0,np.nan),axis=0), counts

def percent_axis(fig, axis='y'):
    getattr(fig, f'update_{axis}axes')(tickformat='.2%', range=[0,1])
    return fig

def heatmap(rates, counts, title, colorscale):
    text = [[pct(v) for v in row] for row in rates.to_numpy()]
    # A gray background plus explicit No data annotations keeps absent cells distinct from 0%.
    fig = go.Figure(go.Heatmap(z=rates.to_numpy(), x=[str(c) for c in rates.columns],
        y=[str(c) for c in rates.index], zmin=0,zmax=1,colorscale=colorscale,
        customdata=counts.to_numpy(), text=text, hoverongaps=False,
        hovertemplate='%{y} / %{x}<br>Satisfaction: %{text}<br>Records: %{customdata:d}<extra></extra>',
        colorbar=dict(tickformat='.2%',title='Satisfied')))
    for i,row in enumerate(text):
        for j,label in enumerate(row):
            value = rates.iloc[i,j]
            fig.add_annotation(x=str(rates.columns[j]),y=str(rates.index[i]),text=label,showarrow=False,
                               font=dict(size=11,color='white' if pd.notna(value) and value>.65 else '#17334f'))
    fig.update_layout(title=title, xaxis_title=rates.columns.name, yaxis_title=rates.index.name,
                      plot_bgcolor='#eef2f6',height=390)
    fig.update_yaxes(autorange='reversed',type='category')
    fig.update_xaxes(type='category')
    return fig

def flight_figures(df):
    stats = delay_summary(df).reset_index()
    line = go.Figure(go.Scatter(x=stats[GROUP],y=stats['rate'],mode='lines+markers+text',
        text=[pct(x) for x in stats['rate']],textposition='top center',line=dict(color=YELLOW,width=3),
        customdata=stats[['count']],connectgaps=False,
        hovertemplate='%{x}<br>Satisfied: %{y:.2%}<br>Records: %{customdata[0]:d}<extra></extra>'))
    line.update_layout(title='Satisfaction by arrival delay group',xaxis_title=GROUP,yaxis_title='Satisfaction rate',height=370)
    percent_axis(line)
    for _,r in stats[stats['rate'].isna()].iterrows():
        line.add_annotation(x=r[GROUP],y=.04,text='No data',showarrow=False)
    # Collapse identical coordinates, retaining every record through counts; no sampling.
    points = df.groupby([DEPARTURE,ARRIVAL],dropna=False).size().reset_index(name='Records')
    scatter = px.scatter(points,x=DEPARTURE,y=ARRIVAL,hover_data={'Records':':d'},
                         color_discrete_sequence=['#c48dbb'],title='Departure vs arrival delay',render_mode='webgl')
    scatter.update_traces(marker=dict(size=5,opacity=.55))
    scatter.update_layout(height=370)
    work = df.assign(Status=df[TARGET].map(LABELS))
    box = px.box(work,x='Status',y=DEPARTURE,color='Status',color_discrete_map=PURPLE,
        category_orders={'Status':list(LABELS.values())},points='outliers',title='Departure delay by satisfaction')
    box.update_layout(showlegend=False,xaxis_title='Satisfaction status',height=400,
                      title='Departure delay<br>by satisfaction')
    box.update_xaxes(tickvals=list(LABELS.values()),ticktext=['Dissatisfied','Satisfied'])
    delayed = df.loc[df[ARRIVAL]>30,TARGET].value_counts().reindex([0,1],fill_value=0)
    pie = go.Figure(go.Pie(labels=list(LABELS.values()),values=delayed.to_numpy(),hole=.45,
        marker_colors=list(PURPLE.values()),sort=False,texttemplate='%{percent:.2%}',
        hovertemplate='%{label}<br>Records: %{value:d}<br>Share: %{percent:.2%}<extra></extra>'))
    pie.update_layout(title='Satisfaction among passengers<br>with arrival delay >30 minutes',height=400,legend=dict(orientation='h',y=-.15))
    if delayed.sum()==0:
        pie.add_annotation(text='No data',x=.5,y=.5,showarrow=False)
    travel = df.groupby('Type of Travel')[DEPARTURE].agg(mean='mean',count='count').reset_index()
    bars = px.bar(travel,x='Type of Travel',y='mean',text_auto='.2f',color_discrete_sequence=[YELLOW],
        title='Average departure delay by travel type',custom_data=['count'])
    bars.update_traces(hovertemplate='%{x}<br>Average delay: %{y:.2f} min<br>Records: %{customdata[0]:d}<extra></extra>')
    bars.update_layout(title='Average departure delay<br>by travel type',yaxis_title='Average departure delay (minutes)',height=400)
    return [line,scatter,box,pie,bars]

def service_figures(df, exclude_zero=False):
    work = df.copy()
    if exclude_zero:
        work['Online boarding'] = work['Online boarding'].replace(0,np.nan)
    boarding = work.groupby('Type of Travel')['Online boarding'].agg(mean='mean',count='count').reset_index()
    board = px.bar(boarding,y='Type of Travel',x='mean',orientation='h',text_auto='.2f',
        color_discrete_sequence=['#ffbf80'],custom_data=['count'],title='Average online boarding rating by travel type')
    board.update_traces(hovertemplate='%{y}<br>Mean rating: %{x:.2f}<br>Rated records: %{customdata[0]:d}<extra></extra>')
    board.update_layout(title='Average online boarding rating<br>by travel type',xaxis_title='Average online boarding rating',yaxis_title=None,height=320)
    board.update_xaxes(range=[0,5])
    for _,r in boarding[boarding['mean'].isna()].iterrows():
        board.add_annotation(x=0,y=r['Type of Travel'],text='No data',showarrow=False,xanchor='left')
    wifi = df['Inflight wifi service'].value_counts().reindex(range(6),fill_value=0)
    wifi_labels = ['0 · Not applicable']+[str(i) for i in range(1,6)]
    wi = go.Figure(go.Bar(x=wifi_labels,y=wifi.to_numpy(),text=wifi.to_numpy(),texttemplate='%{text:d}',
        marker_color=['#b9dff1','#9fcbea','#ffcc99','#ffb366','#8fbfe0','#ff9f40'],
        customdata=(wifi/max(len(df),1)).to_numpy(),
        hovertemplate='Wi-Fi rating: %{x}<br>Records: %{y:d}<br>Share: %{customdata:.2%}<extra></extra>'))
    wi.update_layout(title='Inflight Wi-Fi rating distribution',xaxis_title='Wi-Fi rating',yaxis_title='Passengers',height=300)
    grid_df = df.assign(**{GROUP:delay_groups(df[ARRIVAL])})
    rates,counts = satisfaction_grid(grid_df,'On-board service',GROUP,list(range(6)),DELAY_GROUPS)
    onboard = heatmap(rates,counts,'Satisfaction · on-board service × arrival delay',[[0,'#ffcf9f'],[.35,'#e7e7e7'],[1,'#b9d2ef']])
    onboard.update_layout(title='Satisfaction rate<br>On-board service × arrival delay',height=420)
    onboard.update_xaxes(tickvals=DELAY_GROUPS,ticktext=['No delay','1–30<br>minutes','31–60<br>minutes','61–120<br>minutes','Over 120<br>minutes'],tickangle=0)
    rates,counts = satisfaction_grid(df,'Leg room service','Seat comfort',list(range(6)),list(range(6)))
    comfort = heatmap(rates,counts,'Satisfaction · leg room × seat comfort',[[0,'#e7e7e7'],[.5,'#f4d8c1'],[1,'#ffbf80']])
    comfort.update_layout(title='Satisfaction rate<br>Leg room service × seat comfort',height=420)
    for figure in (onboard, comfort):
        for annotation in figure.layout.annotations:
            annotation.font.color = '#17334f'
    means = mean_ratings(df,exclude_zero)
    dumbbell = go.Figure()
    for aspect in ASPECTS:
        vals = means.loc[aspect,'mean']
        dumbbell.add_trace(go.Scatter(x=vals.to_numpy(),y=[aspect,aspect],mode='lines',
            line=dict(color='#aab7c2',width=2),showlegend=False,hoverinfo='skip'))
    for label,color in [(0,SERVICE_COLORS[LABELS[0]]),(1,SERVICE_COLORS[LABELS[1]])]:
        part = means.xs(label,level=TARGET).reindex(ASPECTS)
        dumbbell.add_trace(go.Scatter(x=part['mean'],y=ASPECTS,mode='markers+text',name=LABELS[label],
            marker=dict(color=color,size=9),text=[f'{x:.2f}' if pd.notna(x) else 'No data' for x in part['mean']],
            textposition='top left' if label==0 else 'bottom right',customdata=part[['count']].fillna(0),
            hovertemplate='%{y}<br>Mean: %{x:.2f}<br>Rated records: %{customdata[0]:d}<extra>%{fullData.name}</extra>'))
        for aspect in part.index[part['mean'].isna()]:
            dumbbell.add_annotation(x=0,y=aspect,text=f'{LABELS[label]}: No data',showarrow=False,
                yshift=10 if label==0 else -10,xanchor='left',font=dict(size=10))
    dumbbell.update_layout(title='Service ratings by satisfaction',xaxis_title='Average service rating',
        height=400,legend=dict(orientation='h',y=-.25),margin=dict(l=140))
    dumbbell.update_xaxes(range=[0,5]); dumbbell.update_yaxes(autorange='reversed')
    rates,counts = class_composition(df)
    stacked = go.Figure()
    for label in [1,0]:
        stacked.add_trace(go.Bar(x=CLASSES,y=rates[label],name=LABELS[label],marker_color=SERVICE_COLORS[LABELS[label]],
            text=[pct(v) if pd.notna(v) else '' for v in rates[label]],textposition='inside',
            customdata=np.column_stack([counts[label],counts.sum(axis=1)]),
            hovertemplate='%{x}<br>%{fullData.name}: %{y:.2%}<br>Records: %{customdata[0]:d}<br>Cabin total: %{customdata[1]:d}<extra></extra>'))
    for cabin in counts.index[counts.sum(axis=1)==0]:
        stacked.add_annotation(x=cabin,y=.5,text='No data',showarrow=False)
    stacked.update_layout(title='Satisfaction composition<br>within cabin class',barmode='stack',
        xaxis_title='Class',yaxis_title='Passengers within class',height=400,legend=dict(orientation='h',y=-.25))
    percent_axis(stacked)
    return [board,wi,onboard,comfort,dumbbell,stacked]

def model_details(predictions, name):
    y,p = predictions['actual'],predictions[name]
    return {'Accuracy':accuracy_score(y,p), 'Actual satisfaction rate':y.mean(),
        'F1-score (Satisfied)':f1_score(y,p,zero_division=0),
        'Precision (Dissatisfied)':precision_score(y,p,pos_label=0,zero_division=0),
        'Precision (Satisfied)':precision_score(y,p,pos_label=1,zero_division=0),
        'Predicted satisfaction rate':p.mean(),
        'Recall (Dissatisfied)':recall_score(y,p,pos_label=0,zero_division=0),
        'Recall (Satisfied)':recall_score(y,p,pos_label=1,zero_division=0)}

def accuracy_figure(report):
    table = pd.DataFrame(report['models']).set_index('model').reindex(MODEL_ORDER).reset_index()
    fig = px.bar(table,y='model',x='accuracy',orientation='h',text_auto='.2%',
        color_discrete_sequence=['#4e7ca8'],title='Held-out test accuracy',custom_data=['model'])
    fig.update_traces(hovertemplate='%{y}<br>Accuracy: %{x:.2%}<extra></extra>')
    fig.update_layout(xaxis_title='Prediction accuracy',yaxis_title=None,height=260)
    fig.update_yaxes(autorange='reversed'); percent_axis(fig,'x')
    return fig

def confusion_figure(matrix,name):
    fig = go.Figure(go.Heatmap(z=matrix,x=list(LABELS.values()),y=list(LABELS.values()),
        colorscale=[[0,'#c1e4dd'],[.5,'#4196a7'],[1,'#2c547d']],texttemplate='%{z:d}',
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Records: %{z:d}<extra></extra>',
        colorbar=dict(title='Records',tickformat='d')))
    fig.update_layout(title=f'{name} · held-out confusion matrix',xaxis_title='Predicted class',yaxis_title='Actual class',height=440)
    fig.update_xaxes(tickvals=list(LABELS.values()),ticktext=['Dissatisfied','Satisfied'])
    fig.update_yaxes(tickvals=list(LABELS.values()),ticktext=['Dissatisfied','Satisfied'])
    fig.update_yaxes(autorange='reversed')
    return fig
