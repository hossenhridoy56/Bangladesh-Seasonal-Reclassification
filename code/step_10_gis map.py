import os
import folium
from folium.plugins import MarkerCluster

script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(script_dir, ".."))
fig_path  = os.path.join(base_path, "figures")

os.makedirs(fig_path, exist_ok=True)

station_coords = {
    'Ambaganctg': [22.3500, 91.8200], 'Barisal': [22.7010, 90.3535],
    'Bhola': [22.6859, 90.6440], 'Bogra': [24.8481, 89.3730],
    'Chandpur': [23.2321, 90.6631], 'Chuadanga': [23.6401, 88.8414],
    'Comilla': [23.4607, 91.1809], 'Conavg': [23.8500, 90.4000],
    'Coxs Bazar': [21.4272, 91.9702], 'Dhaka': [23.8103, 90.4125],
    'Dinajpur': [25.6217, 88.6354], 'Faridpur': [23.6071, 89.8429],
    'Feni': [23.0159, 91.3976], 'Hatiya': [22.2855, 91.1174],
    'Ishurdi': [24.1285, 89.0645], 'Jessore': [23.1667, 89.2167],
    'Khepupara': [21.8311, 90.1265], 'Khulna': [22.8456, 89.5403],
    'Kutubdia': [21.8154, 91.8540], 'Madaripur': [23.1641, 90.1833],
    'Maijdi Court': [22.8614, 91.0991], 'Mongla': [22.4764, 89.5981],
    'Mymensingh': [24.7471, 90.4203], 'Patangactg': [22.2300, 91.8000],
    'Patuakhali': [22.3524, 90.3346], 'Rajshahi': [24.3745, 88.6042],
    'Rangamati': [22.6333, 92.2000], 'Rangpur': [25.7439, 89.2752],
    'Sandwip': [22.4844, 91.4322], 'Satkhira': [22.7073, 89.0720],
    'Sayedpur': [25.7500, 88.9167], 'Sitakunda': [22.6157, 91.6592],
    'Srimangal': [24.3083, 91.7333], 'Sylhet': [24.8949, 91.8687],
    'Tangail': [24.2513, 89.9167], 'Teknaf': [20.8583, 92.2978]
}

m = folium.Map(location=[23.6850, 90.3563], zoom_start=7, tiles='CartoDB positron')
cluster = MarkerCluster().add_to(m)

for name, coord in station_coords.items():
    folium.CircleMarker(
        location=coord,
        radius=6,
        color='#2166AC',
        fill=True,
        fill_opacity=0.9,
        popup=name
    ).add_to(cluster)

title_html = """
<h3 align="center"><b>
Figure 1: Spatial Distribution of 36 Meteorological Stations in Bangladesh
</b></h3>
<p align="center">
<a href="https://htmlpreview.github.io/?https://github.com/hossenhridoy56/Bangladesh-Seasonal-Reclassification/blob/main/figures/Figure1_Map.html" target="_blank">
🗺️ Click here to view Interactive Map
</a>
</p>
"""
m.get_root().html.add_child(folium.Element(title_html))

html_output = os.path.join(fig_path, "Figure1_Map.html")
m.save(html_output)

print(f"Map successfully saved in: {html_output}")