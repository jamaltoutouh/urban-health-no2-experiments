import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import pandas as pd
from statistics import mean, stdev


# Read data from CSV
day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
hours = list(range(24))

data_path = '../data/csv/035-08-sequence_air_all.csv'
data = pd.read_csv(data_path)

# For the first graph
week_data = dict()
for i, day  in enumerate(day_name):
    week_data[day] = []
    for j in range(24):
        measures = list(data[(data['weekday'] == i) & (data['hour'] == j)]['measure'])
        week_data[day].append(mean(measures))

# For the second graph
workingday_data = dict()
hours_rev = hours[::-1]
for is_working in [True, False]:
    label = 'Working day' if is_working else 'Weekend'
    workingday_data[label] = dict()
    workingday_data[label]['mean'] = []
    workingday_data[label]['max'] = []
    workingday_data[label]['min'] = []
    for j in range(24):
        measures = list(data[(data['working-day'] == is_working) & (data['hour'] == j)]['measure'])
        sstedv = stdev(measures)
        mmean = mean(measures)
        workingday_data[label]['mean'].append(mmean)
        workingday_data[label]['max'].append(mmean + sstedv)
        workingday_data[label]['min'].append(mmean - sstedv)
    workingday_data[label]['min'] = workingday_data[label]['min'][::-1]

# Create lines
dashes = ['solid', 'dashdot', 'dash', 'dot']
whole_fig = make_subplots(rows=2, cols=1)

# Weekdays
for i, day in enumerate(day_name):
    if i < 5:
        print(i%4)
        whole_fig.add_trace(go.Scatter(x=hours, y=week_data[day], name=day,
                             line=dict(color='rgb(0, {}, {})'.format(0, 255-40*i), dash=dashes[i%4], width=4)), row=1, col=1)
    else:
        whole_fig.add_trace(go.Scatter(x=hours, y=week_data[day], name=day,
                                 line=dict(color='rgb({}, {}, 0)'.format(255-40*(i%5), 0), dash=dashes[i%4], width=4)), row=1, col=1)

for label in ['Working day', 'Weekend']:
    colors = ['rgba(255,0,0,0.1)','rgba(148, 0, 0,1)'] if label == 'Weekend' else ['rgba(0,0,255,0.1)',' rgba(0, 11, 163,1)']
    whole_fig.add_trace(go.Scatter(
        x=hours+hours_rev,
        y=workingday_data[label]['max']+workingday_data[label]['min'],
        fill='toself',
        fillcolor=colors[0],
        line_color='rgba(0, 0, 0,0)',
        showlegend=False,
        name=label,
    ), row=2, col=1)
    whole_fig.add_trace(go.Scatter(x=hours, y=workingday_data[label]['mean'], name=label,
                                   showlegend=True,
                                   line=dict(color=colors[1], dash='solid',
                                             width=4)), row=2, col=1)



# Edit the layout
whole_fig.update_layout(title='Average NO2 concentration',
                        xaxis_title='Hour of the day',
                        yaxis_title='NO2 concentration',
                        xaxis2_title='Hour of the day',
                        yaxis2_title='NO2 concentration'
                        )

whole_fig.update_layout(
    xaxis2 = dict(
        tickmode = 'array',
        tickvals = hours,
        ticktext = hours
    ),
    xaxis = dict(
        tickmode = 'array',
        tickvals = hours,
        ticktext = hours
    )
)
whole_fig.update_layout(
    yaxis=dict(tickmode = 'array',
        tickvals = [0, 20, 40, 60, 80],
        ticktext = [0, 20, 40, 60, 80],
        range=[10, 80]
    ),
    yaxis2=dict(tickmode = 'array',
        tickvals = [0, 20, 40, 60, 80],
        ticktext = [0, 20, 40, 60, 80],
        range=[10, 80]
    )
)

whole_fig.show()
pio.write_html(whole_fig, file='index.html', auto_open=True)