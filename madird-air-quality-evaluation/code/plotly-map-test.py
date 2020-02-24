import plotly.graph_objects as go
import pandas as pd
import random
import plotly.io as pio

import sys

mapbox_access_token = open(".mapbox_token").read()
stations = pd.read_csv('../stations-information.csv', header=0)

# Remove bad sensors
stations = stations[stations['Id'] != 27]
stations = stations[stations['Id'] != 59]

lons = [str(x) for x in list(stations['Longitud'])]
lats = [str(x) for x in list(stations['Latitud'])]
gaps = [float(x) for x in list(stations['GAP'])]
names = ['{} ({}) reduction={:.1f}%'.format(station, id, -gap) for station, id, gap in zip(list(stations['Estacion']), list(stations['Id']), gaps)]
ids = ['{}'.format(id) for id in list(stations['Id'])]

def get_color_gap(gap, threshold):
    colors = ['rgb(154, 50, 50)', 'rgb(154, 76, 50)', 'rgb(154, 102, 50)', 'rgb(154, 128, 50)', 'rgb(154, 154, 50)', 'rgb(128, 154, 50)', 'rgb(102, 154, 50)', 'rgb(76, 154, 50)', 'rgb(50, 154, 50)', 'rgb(50, 154, 76)']
    colors = ['rgb(154, 50, 50)', 'rgb(154, 154, 50)',
              'rgb(128, 154, 50)', 'rgb(102, 154, 50)', 'rgb(76, 154, 50)', 'rgb(50, 154, 50)', 'rgb(50, 154, 76)']
    if gap >= 0:
        return 'rgb(0, 0, 0)'
        return colors[0]
    index = int(-gap/threshold * len(colors))
    if index == 0:
        index += 1
    return 'rgb(0, 0, 0)'
    return colors[index]

colors = [get_color_gap(gap, 25) for gap in gaps]
print(colors)



fig = go.Figure()



fig = go.Figure(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=15,
            color=colors
        ),
        text=ids,
        textposition = "middle right",
        marker_symbol = 'circle'
    ))

fig.update_layout(
    autosize=True,
    hovermode='closest',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=40.420017,
            lon=-3.7077736
        ),
        pitch=0,
        zoom=10,
        style='light'
    ),
)

pio.write_html(fig, file='index.html', auto_open=True)
