
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

def aggregate_data(directory):
    all_data = []
    latitude, longitude, topology = [], [], []
    # Iterate through all CSV files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            data, lat, long, topol = process_csv(file_path)
            all_data.append(data)
            latitude.append(lat)
            longitude.append(long)
            topology.append(topol)

            # break
    
    # Concatenate all data into a single DataFrame
    combined_data = np.stack(all_data).transpose(1,0,2)
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    topology = np.array(topology)
    extra_info = np.stack([latitude, longitude, topology])
    return combined_data, extra_info

# %%
# Directory containing the CSV files
csv_directory = os.getenv("CSV_DIRECTORY")

# Aggregate all data
combined_data, extra_info = aggregate_data(csv_directory)
print(combined_data.shape, extra_info.shape)

# %%
combined_data_img = combined_data.reshape(-1, 34, 23, 2)
combined_data_img = combined_data_img.transpose(0,2,1,3)

#%%
extra_info_img = extra_info.reshape(-1, 34, 23)
extra_info_img = extra_info_img.transpose(0,2,1)
#%%


np.save('tensor.npy', combined_data_img, allow_pickle=True)
np.save('lat_long_top.npy', extra_info_img, allow_pickle=True)

# %%
