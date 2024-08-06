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
    return wind_data[['Wind Speed','Wind Direction']].to_numpy()

def aggregate_data(directory):
    all_data = []
    
    # Iterate through all CSV files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            data = process_csv(file_path)
            all_data.append(data)
            # break
    
    # Concatenate all data into a single DataFrame
    combined_data = np.stack(all_data).transpose(1,0,2)
    
    return combined_data

# %%
# Directory containing the CSV files
csv_directory = os.getenv("CSV_DIRECTORY")

# Aggregate all data
combined_data = aggregate_data(csv_directory)
print(combined_data)

# %%
combined_data_img = combined_data.reshape(-1, 34, 23, 2)
combined_data_img = combined_data_img.transpose(0,2,1,3)


np.save('tensor.npy', combined_data_img, allow_pickle=True)

# %%
# import numpy as np
# from PIL import Image
# import os

# # Create directories to save images if they don't exist
# if not os.path.exists('data'):
#     os.makedirs('data')

# if not os.path.exists('data/all_images'):
#     os.makedirs('data/all_images')

# if not os.path.exists('data/one_half_images'):
#     os.makedirs('data/one_half_images')

# # Function to save a single image
# def save_image(image_array, filename):
#     image = Image.fromarray(image_array)
#     image.save(filename)

# allmax = np.max(combined_data_img[:,:,:,0])
# allmin = np.min(combined_data_img[:,:,:,0])
# #%%
# # Save all images
# for i in range(combined_data_img.shape[0]):
#     # Process the first channel as an example
#     img1 = combined_data_img[i, :, :, 0]
#     # Convert to uint8
#     img1 = (255 * (img1 - allmin) / (allmax - allmin)).astype(np.uint8)
#     save_image(img1, f'data/all_images/{i+1}.png')
#     # save_image(img1, f'all_images/image_{i:05d}.png')

#     # Optionally save only 1 out of every 2 images
#     if i % 2 == 0:
#         save_image(img1, f'data/one_half_images/{i//2+1}.png')
#     if i > 1000:
#         break
    
# # %%
# import json

# # Save the min and max values in a JSON file
# metadata = {'min': float(allmin), 'max': float(allmax)}
# with open('data/metadata.json', 'w') as json_file:
#     json.dump(metadata, json_file)

# # %%
