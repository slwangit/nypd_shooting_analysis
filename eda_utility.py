
import pandas as pd
import geopandas as gpd
import seaborn as sns
import plotly.express as px
import folium
from folium import plugins
import matplotlib.pyplot as plt


def month_plotting(df):
    # Exploring month
    # Group by month
    month_trend = df.groupby(['year', 'month']).count().reset_index().iloc[:, 0:3].rename(
        columns={'OCCUR_DATE': 'Shooting Cases'})

    # Plot month trend by year
    month_plot = px.bar(month_trend, x="month", y="Shooting Cases", color='year', barmode='group')
    month_plot.update_layout(title='Monthly Shooting Incidents by Year',
                             xaxis_title='Month',
                             yaxis_title='Total Incidents')

    month_plot.show()


def time_weekday_plotting(df):
    # Exploring time period and weekday frequency
    # Group by time period(Only focus on recent 3 years)
    tmp = df[df['year'].isin([2019, 2020, 2021])]
    time_trend = tmp.groupby('OCCUR_TIME').count().reset_index().iloc[:, 0:2].rename(
        columns={'OCCUR_DATE': 'Shooting Cases'})

    # Group by weekday(Only focus on recent 3 years)
    weekday_trend = tmp.groupby('weekday').count().reset_index().iloc[:, 0:2].rename(
        columns={'OCCUR_DATE': 'Shooting Cases'})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Shooting Frequency in the Recent 3 Years')

    sns.barplot(ax=ax1, x="OCCUR_TIME", y="Shooting Cases", data=time_trend)
    sns.barplot(ax=ax2, x="weekday", y="Shooting Cases", data=weekday_trend)

    ax1.set(xlabel='Time Period', ylabel='Total Incidents', title='By Time Period')
    ax2.set(xlabel='Weekday', ylabel='Total Incidents', title='By Weekday')

    fig.subplots_adjust(wspace=0.2)

    plt.show()


def age_distribution(df):
    # Victims age distribution
    vic_fig = px.histogram(df, x="VIC_AGE_GROUP",
                           color="VIC_SEX",
                           barmode="group",
                           category_orders={'VIC_AGE_GROUP': ['<18', '18-24', '25-44', '45+']})

    # Perpetrator age distribution
    perp_fig = px.histogram(df.dropna(), x="PERP_AGE_GROUP",
                            color="PERP_SEX",
                            barmode="group")

    vic_fig.update_layout(title='Age distribution of Victims', title_x=0.5,
                          xaxis_title='Age Group')
    perp_fig.update_layout(title='Age distribution of Perpetrators', title_x=0.5,
                           xaxis_title='Age Group')
    vic_fig.show()
    perp_fig.show()


def hotspot_map(df, location):
    # In case of overplotting, limited to shooting incidents in the recent 5 years
    tmp = df[df.year >= 2016]

    # Instantiate a feature group for the incidents in the dataframe and a folium map
    incidents = folium.map.FeatureGroup()
    hotspot_map = folium.Map(location=location, zoom_start=11)

    # Loop through the crimes and add each to the incidents feature group
    for lat, lng, in zip(tmp.Latitude, tmp.Longitude):
        incidents.add_child(
            folium.CircleMarker(
                location=[lat, lng],
                radius=1,  # define how big you want the circle markers to be
                color='red',
                fill=True
            )
        )

    # Add incidents to map
    hotspot_map.add_child(incidents)

    return hotspot_map


def cluster_map(df, location):
    # Instantiate a folium map
    cluster_map = folium.Map(location=location, zoom_start=11)

    # instantiate a mark cluster object for the incidents in the dataframe
    cluster = plugins.MarkerCluster()

    # loop through the dataframe and add each data point to the mark cluster
    for lat, lng, in zip(df.Latitude, df.Longitude):
        folium.Marker(
            location=[lat, lng],
            icon=None,
        ).add_to(cluster)

    # add incidents to map
    cluster_map.add_child(cluster)

    return cluster_map


def choropleth_map(df, location, borogeo):
    boro = pd.DataFrame(df['BORO'].value_counts()).reset_index().rename(
        columns={'index': 'Neighborhood', 'BORO': 'Count'})

    boro['Neighborhood'] = boro['Neighborhood'].str.capitalize()
    boro.iloc[4, 0] = 'Staten Island'

    choropleth_map = folium.Map(location=location, zoom_start=10)

    folium.Choropleth(
        geo_data=borogeo,
        data=boro,
        columns=['Neighborhood', 'Count'],
        key_on='feature.properties.boro_name',
        fill_color='PuRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        highlight=True,
        legend_name='Shooting Frequency in New York City'
    ).add_to(choropleth_map)

    return choropleth_map


def main(df):

    # EDA
    # Exploring month frequency
    month_plotting(df)

    # Exploring time period and weekday frequency
    time_weekday_plotting(df)

    # age and sex distribution
    age_distribution(df)

    nyc_location = [40.71, -74.00]

    # maps
    hots_map = hotspot_map(df, nyc_location)

    clus_map = cluster_map(df, nyc_location)

    # import borough data
    city = gpd.read_file('files/Borough_Boundaries.geojson')

    choro_map = choropleth_map(df, nyc_location, city)


if __name__ == '__main__':
    main()
