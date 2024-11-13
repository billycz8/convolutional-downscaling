
# %%
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract header information
    header_info = df.iloc[0:1, :].to_dict(orient='records')[0]
    
    # Extract wind data
    wind_data = df.iloc[1:, :].copy()
    wind_data.columns = df.iloc[1, :]
    wind_data = wind_data.iloc[1:, :].reset_index(drop=True)
    
    # Convert columns to appropriate types
    # wind_data[['Year', 'Month', 'Day', 'Hour', 'Minute']] = wind_data[['Year', 'Month', 'Day', 'Hour', 'Minute']].astype(int)
    wind_data['Wind Speed'] = wind_data['Wind Speed'].astype(float)
    wind_data['Wind Direction'] = wind_data['Wind Direction'].astype(int)
    return wind_data[['Wind Speed','Wind Direction']].to_numpy(), float(header_info['Latitude']), float(header_info['Longitude']), float(header_info['Elevation'])

def aggregate_data(base_directory):
    
    combined_year_data = []
    latitude, longitude, topology = [], [], []
    # Iterate through each subdirectory in the base directory
    first_year = True
    for subdir in sorted(os.listdir(base_directory)):
        subdir_path = os.path.join(base_directory, subdir)
        # if subdir in ["2019","2020","2021"]:
        #     continue
        # Only process subdirectories (e.g., 2019, 2020)
        if os.path.isdir(subdir_path):
            year_data = []
            for filename in sorted(os.listdir(subdir_path)):
                if filename.endswith(".csv"):
                    print(filename)
                    file_path = os.path.join(subdir_path, filename)
                    data, lat, long, topol = process_csv(file_path)

                     # Exclude the unwanted boundary data
                    if lat != 44.08 and long != -82.01:
                        year_data.append(data)
                        if first_year:
                            latitude.append(lat)
                            longitude.append(long)
                            topology.append(topol)

            # combined_year_data.append(np.stack(year_data).transpose(1,0,2))
            year_data_stacked = np.stack(year_data).transpose(1,0,2)
            combined_data_img = year_data_stacked.reshape(-1, 32,32, 2)
            combined_data_img = combined_data_img.transpose(0,2,1,3)
            np.save(f'tensor_{subdir}.npy', combined_data_img, allow_pickle=True)
            
            if first_year:
                latitude = np.array(latitude)
                longitude = np.array(longitude)
                topology = np.array(topology)
                extra_info = np.stack([latitude, longitude, topology])
                extra_info_img = extra_info.reshape(-1, 32,32)
                extra_info_img = extra_info_img.transpose(0,2,1)
                np.save('lat_long_top.npy', extra_info_img, allow_pickle=True)
            first_year = False


            # break
    
    # Concatenate all data into a single DataFrame
    # combined_data = np.concatenate(combined_year_data, axis=0)
    # latitude = np.array(latitude)
    # longitude = np.array(longitude)
    # topology = np.array(topology)
    # extra_info = np.stack([latitude, longitude, topology])
    # return combined_data, extra_info

# %%
# Directory containing the CSV files
csv_directory = os.getenv("CSV_DIRECTORY")

# Aggregate all data
aggregate_data(csv_directory)
