import pandas as pd
import numpy as np

def main():
    total_num_users = 182
    max_latitudes = list()
    min_latitudes = list()
    max_longitudes = list()
    min_longitudes = list()
    # Used to identify if some users contain wrong values
    users_to_check = list()

    lat_unique = list()
    lon_unique = list()

    for i in range(total_num_users):
        df = pd.read_csv(f'geolife_geolife_trajectories_user_{i}.csv')

        lat_unique_user = np.unique(df['lat'])
        lon_unique_user = np.unique(df['lon'])

        lat_unique.extend(lat_unique_user.tolist())
        lon_unique.extend(lon_unique_user.tolist())



        max_lat = df['lat'].max()
        min_lat = df['lat'].min()
        max_lon = df['lon'].max()
        min_lon = df['lon'].min()
        max_latitudes.append(max_lat)
        min_latitudes.append(min_lat)
        max_longitudes.append(max_lon)
        min_longitudes.append(min_lon)
        if max_lat > 90 or min_lat < -90 or max_lon > 180 or min_lon < -180:
            users_to_check.append(i)


    lat_unique = list(set(lat_unique))
    lat_unique = [int(lat) for lat in lat_unique]
    lat_unique = list(set(lat_unique))


    lon_unique = list(set(lon_unique))
    lon_unique = [int(lon) for lon in lon_unique]
    lon_unique = list(set(lon_unique))

    print("lat_unique len: ", len(lat_unique))
    print("Latitude unique values: ", lat_unique)

    print("lon_unique len: ", len(lon_unique))
    print("Longitude unique values: ", lon_unique)

    print("Max latitude in dataset: ", max(max_latitudes))
    print("Min latitude in dataset: ", min(min_latitudes))
    print("Max longitude in dataset: ", max(max_longitudes))
    print("Min longitude in dataset: ", min(min_longitudes))
    print("Please check users with indexes: ", users_to_check)


if __name__ == '__main__':
    main()
