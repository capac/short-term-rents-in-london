#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import itertools
import pandas as pd
import folium
from folium.plugins import HeatMap
from matplotlib import colormaps
import matplotlib.colors as colors


home_dir = Path.home()
inside_airbnb_data_dir = home_dir / 'Programming/data/inside-airbnb/london'
inside_airbnb_work_dir = (
    home_dir /
    'Programming/Python/machine-learning-exercises/short-term-rents-in-london'
    )

inside_airbnb_data_file = (
    inside_airbnb_data_dir /
    'selected_short_term_rentals_with_distances.csv'
)
inside_airbnb_df = pd.read_csv(inside_airbnb_data_file,
                               keep_default_na=False, thousands=',')

inside_airbnb_df.drop(['room_type', 'nearest_station'], axis=1, inplace=True)
inside_airbnb_df['borough'] = (
    inside_airbnb_df['borough'].replace({r'\s': r'_'}, regex=True)
    )

london_map_1 = folium.Map(location=[51.5074, -0.1278], zoom_start=12,
                          tiles='CartoDB Voyager')

borough_names = inside_airbnb_df['borough'].unique()
num_locations = len(borough_names)
colormap = list(itertools.chain(
    colormaps['tab20b'].colors,
    colormaps['tab20c'].colors
))
location_colors = {loc: colors.rgb2hex(colormap[i])
                   for i, loc in enumerate(borough_names)}

for _, row in inside_airbnb_df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=4,
        color=location_colors[row['borough']],
        fill=True,
        fill_color=location_colors[row['borough']],
        fill_opacity=0.6,
        weight=0,
    ).add_to(london_map_1)

path_plots = inside_airbnb_work_dir / 'html'
path_plots.mkdir(exist_ok=True, parents=True)
density_map_file = inside_airbnb_work_dir / 'html/density_map.html'
if not density_map_file.exists():
    london_map_1.save(density_map_file)


london_map_2 = folium.Map(location=[51.5074, -0.1278],
                          zoom_start=12, tiles='CartoDB Voyager')

_ = HeatMap(
        inside_airbnb_df[['latitude', 'longitude']].values,
        radius=20,
        blur=10,
        min_opacity=0.2,
        max_opacity=0.8,
).add_to(london_map_2)


heat_map_file = inside_airbnb_data_dir / 'html/heat_map.html'
if not heat_map_file.exists():
    london_map_2.save(inside_airbnb_work_dir / 'html/heat_map.html')
