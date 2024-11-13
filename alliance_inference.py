# %% 
import numpy as np
from sklearn.model_selection import train_test_split
#%%
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


print(tf.__version__)


physical_devices = tf.config.list_physical_devices('GPU')
print(len(physical_devices))

from dotenv import load_dotenv
import os
load_dotenv()
ORIGINAL_RESOLUTION = int(os.getenv("ORIGINAL_RESOLUTION"))
FINAL_RESOLUTION = int(os.getenv("FINAL_RESOLUTION"))
model_res = FINAL_RESOLUTION

def grad_loss(v_gt, v):
    # Gradient loss
    loss = tf.reduce_mean(tf.abs(v - v_gt), axis=[1,2,3])
    jy = v[:,:,1:,:,:] - v[:,:,:-1,:,:]
    jx = v[:,:,:,1:,:] - v[:,:,:,:-1,:]
    jy_ = v_gt[:,:,1:,:,:] - v_gt[:,:,:-1,:,:]
    jx_ = v_gt[:,:,:,1:,:] - v_gt[:,:,:,:-1,:]
    loss += tf.reduce_mean(tf.abs(jy - jy_), axis=[1,2,3])
    loss += tf.reduce_mean(tf.abs(jx - jx_), axis=[1,2,3])
    return loss

def uNet(input, time, lat, lon, height, direction, kernel = [5, 3, 3], nodes = [72, 144, 288, 576]):
    '''
    This function defines a U-Net architecture
    :param input: the main-input layer
    :param time: the time input layer
    :param lat, lon, height: additional fields
    :param kernel: Kernel-sizes (default = [5, 3, 3])
    :param nodes: different neuron-sizes if needed (default = [72, 144, 288, 576])
    :return: last layer of constructed model
    '''

    # set Timesteps
    TS = 3
    ##################################################### 1st Block ####################################################
    conv1 = Conv3D(filters      = nodes[0],
                   kernel_size  = (TS, kernel[0], kernel[0]),
                   activation   = 'relu',
                   padding      = 'same',
                   data_format  = 'channels_last')(input)
    mergetime = Concatenate(axis=4)([conv1, lat, lon, height, direction])
    conv1 = Conv3D(filters      = nodes[0],
                   kernel_size  = kernel[0],
                   activation   = 'relu',
                   padding      = 'same',
                   data_format  = 'channels_last')(mergetime)

    pool1 = MaxPooling3D(pool_size = (1, 2, 2))(conv1)

    ##################################################### 2nd Block ####################################################
    conv2 = Conv3D(filters      = nodes[1],
                   kernel_size  = (TS, kernel[1], kernel[1]),
                   activation   = 'relu',
                   padding      = 'same',
                   data_format  = 'channels_last')(pool1)
    conv2 = Conv3D(filters      = nodes[1],
                   kernel_size  = (TS, kernel[1], kernel[1]),
                   activation   = 'relu',
                   padding      = 'same',
                   data_format  = 'channels_last')(conv2)

    pool2 = MaxPooling3D(pool_size = (1, 2, 2))(conv2)

    ##################################################### 3rd Block ####################################################
    conv3 = Conv3D(filters      = nodes[2],
                   kernel_size  = (TS, kernel[2], kernel[2]),
                   activation = 'relu',
                   padding      = 'same',
                   data_format  = 'channels_last')(pool2)
    conv3 = Conv3D(filters      = nodes[2],
                   kernel_size  = (TS, kernel[2], kernel[2]),
                   activation='relu',
                   padding      = 'same',
                   data_format  = 'channels_last')(conv3)

    pool3 = MaxPooling3D(pool_size = (1, 2, 2))(conv3)

    ##################################################### 4th Block ####################################################
    conv4 = Conv3D(filters      = nodes[3],
                   kernel_size  = (TS, kernel[2], kernel[2]),
                   activation='relu',
                   padding      = 'same',
                   data_format  = 'channels_last')(pool3)
    conv4 = Conv3D(filters      = nodes[3],
                   kernel_size  = (TS, kernel[2], kernel[2]),
                   activation='relu',
                   padding      = 'same',
                   data_format  = 'channels_last')(conv4)

    ####################################################### TIME #######################################################
    # Merge time-layer at this point
    mergetime = Concatenate(axis=4)([conv4, time])

    ################################################### UP 3rd Block ###################################################
    # Up-Size again
    up3   = UpSampling3D(size = (1, 2, 2))(mergetime)
    up3   = Conv3D(filters              = nodes[2],
                   kernel_size          = (TS, kernel[1], kernel[1]),
                   activation           = 'relu',
                   padding              = 'same',
                   kernel_initializer   = 'he_normal')(up3)

    # Skip connection
    merge3 = Concatenate(axis=4)([conv3, up3])

    conv3 = Conv3D(filters              = nodes[2],
                   kernel_size          = (TS, kernel[1], kernel[1]),
                   activation           = 'relu',
                   padding              = 'same',
                   data_format          = 'channels_last')(merge3)
    conv3 = Conv3D(filters              = nodes[2],
                   kernel_size          = (TS, kernel[1], kernel[1]),
                   activation           = 'relu',
                   padding              = 'same',
                   data_format          = 'channels_last')(conv3)

    ################################################### UP 2nd Block ###################################################
    up2 = UpSampling3D(size = (1, 2, 2))(conv3)
    up2 = Conv3D(filters                = nodes[1],
                 kernel_size            = (TS, kernel[1], kernel[1]),
                 activation             = 'relu',
                 padding                = 'same',
                 kernel_initializer     = 'he_normal')(up2)

    # Skip connection
    merge2 = Concatenate(axis=4)([conv2, up2])

    conv2 = Conv3D(filters              = nodes[1],
                   kernel_size          = (TS, kernel[1], kernel[1]),
                   activation           = 'relu',
                   padding              = 'same',
                   data_format          = 'channels_last')(merge2)
    conv2 = Conv3D(filters              = nodes[1],
                   kernel_size          = (TS, kernel[1], kernel[1]),
                   activation           = 'relu',
                   padding              = 'same',
                   data_format          = 'channels_last')(conv2)

    ################################################### UP 1st Block ###################################################
    up1 = UpSampling3D(size = (1, 2, 2))(conv2)
    up1 = Conv3D(filters                = nodes[0],
                 kernel_size            = (TS, kernel[0], kernel[0]),
                 activation             = 'relu',
                 padding                = 'same',
                 kernel_initializer     = 'he_normal')(up1)

    merge1 = Concatenate(axis=4)([conv1, up1])

    conv1 = Conv3D(filters              = nodes[0],
                   kernel_size          = (TS, kernel[0], kernel[0]),
                   activation           = 'relu',
                   padding              = 'same',
                   data_format          = 'channels_last')(merge1)
    conv1 = Conv3D(filters              = nodes[0],
                   kernel_size          = (TS, kernel[0], kernel[0]),
                   activation           = 'relu',
                   padding              = 'same',
                   data_format          = 'channels_last')(conv1)

    # last layer is the output
    output = conv1

    return output

def get_model(PS=32, loss = grad_loss, optimizer = 'adam', nodes = [72, 144, 288, 576], residual = False):
    '''
    This function creates the DCN-architecture (residual = False) or RPN-architecture (residual = True).
    :param PS: Patch size
    :param loss: loss function (default = grad_loss)
    :param optimizer: optimizer (default = 'adam')
    :param nodes: different neuron-sizes if needed (default = [72, 144, 288, 576])
    :param residual: boolean toggeling between RPN (True) and DCN (False)
    :return: Model
    '''

    # Input layers
    main_input  = Input(shape = (3, PS, PS, 1))
    time        = Input(shape = (3, int(PS/8), int(PS/8), 1))
    lat         = Input(shape = (3, PS, PS, 1))
    lon         = Input(shape = (3, PS, PS, 1)) 
    height      = Input(shape = (3, PS, PS, 1))
    direction   = Input(shape = (3, PS, PS, 1))

    # Load U-Net
    unet        = uNet(main_input, time, lat, lon, height, direction, nodes = nodes)

    # Define output layer after U-Net
    temp_out    = Conv3D(filters        = 1,
                         kernel_size    = (3, 1, 1),
                         activation     = 'linear',
                         padding        = 'same',
                         data_format    = "channels_last")(unet)

    def transpose_fn(x):
      return tf.transpose(x, perm=[0, 4, 2, 3, 1])
    temp_out = Lambda(transpose_fn)(temp_out)
    # # Define output layer after U-Net
    temp_out    = Conv3D(filters        = 2,
                         kernel_size    = (1, 1, 1),
                         activation     = 'linear',
                         padding        = 'valid',
                         data_format    = "channels_last")(temp_out)
    temp_out = Lambda(transpose_fn)(temp_out)
    
    # residual layer
    if residual:
        temp_out = Add()([main_input[:,1,:,:], temp_out])

    # create model with the defined Layers
    model       = Model(inputs          = [main_input, time, lat, lon, height, direction],
                        outputs         = temp_out)

    # compile with defined loss and optimizer
    model.compile(loss      = loss,
                  optimizer = optimizer,
                  metrics   = ['mse', 'mae', 'mape'])

    return model

# %%
# Load data

# List of years to load
# years = [2019, 2020, 2021, 2022]
year = 2022
years = [year]
# Initialize a list to store data for each year
data_list = []

# Loop over each year and load the corresponding data
for year in years:
    file_name = f'tensor_{year}.npy'
    year_data = np.load(file_name)  # Load data for the specific year
    print(f'Shape of {file_name} is {year_data.shape}')
    data_list.append(year_data)

# Concatenate all data along the first axis
wind_speed_data = np.concatenate(data_list, axis=0)
wind_speed_data_point = wind_speed_data[:,15,15,0]

# wind_speed_data = np.load('tensor_2020.npy')  # Assuming wind speed is in the first channel
print(f'Shape of wind speed data is {wind_speed_data.shape}')
wind_direction_data = wind_speed_data[:, :, :, 1]  # Extract wind direction channel
wind_speed_data = wind_speed_data[:, :, :, 0]  # Extract wind speed channel

import json
# Calculate min and max for wind speed and wind direction
wind_speed_min = np.min(wind_speed_data)
wind_speed_max = np.max(wind_speed_data)

# Create a dictionary to store the min and max values
normalization_params = {
    "wind_speed_min": wind_speed_min,
    "wind_speed_max": wind_speed_max,
}

# Save the dictionary to a JSON file
with open('normalization_params.json', 'w') as f:
    json.dump(normalization_params, f)

print(f"Normalization parameters saved: {normalization_params}")

# Normalize the data
wind_speed_data = (wind_speed_data - np.min(wind_speed_data)) / (np.max(wind_speed_data) - np.min(wind_speed_data))
wind_direction_data = (wind_direction_data - np.min(wind_direction_data)) / (np.max(wind_direction_data) - np.min(wind_direction_data))

print(f'Shape of normalized wind speed data is {wind_speed_data.shape}')
print(f'Shape of normalized wind direction data is {wind_direction_data.shape}')


# Load data
lat_long_top = np.load('lat_long_top.npy')  # Assuming wind speed is in the first channel
for l in range(lat_long_top.shape[0]):
    lat_long_top[l] = (lat_long_top[l] - np.min(lat_long_top[l])) / (np.max(lat_long_top[l]) - np.min(lat_long_top[l]))



# Reshape data to fit the model input shape
def reshape_data(data, target_shape=(32, 32)):
    # Reshape data to the target shape
    data = data[..., np.newaxis]
    # with tf.device('/CPU:0'):
    #     # resized_data = tf.image.resize(data, target_shape, method='bilinear')
    resized_data = tf.image.resize(data, target_shape, method='bilinear')
    data = resized_data.numpy()
    return data

def create_patches_inference(data, time_steps):
    window_size = time_steps // 2 + 1
    num_samples = data.shape[0] - window_size + 1
    training_patches = []
    patches = []

    for i in range(num_samples):
        patch = data[i:i+window_size]
        training_patches.append(patch)
    
    training_timestep = time_steps//2+1
    relative_time = np.zeros((time_steps//2+1, data.shape[1]//8, data.shape[2]//8, 1))

    # Assign specific values to each slice along the first axis
    half_range = (training_timestep // 2) * 2
    time_values = np.arange(-half_range, half_range + 1, 2) * model_res
    for t in range(training_timestep):
        relative_time[t, :, :, 0] = time_values[t]
    relative_time_patches = np.tile(relative_time, (num_samples, 1, 1, 1, 1))
    training_patches = np.array(training_patches)
    patches = np.array(patches)

    return training_patches, relative_time_patches, patches

# import pdb; pdb.set_trace()
# Parameters
patch_size = 32
time_steps_label = 5 # means [0 2 4] for training [1 3] as label

# Create label patches with time_steps_label
wind_speed_data = np.expand_dims(wind_speed_data, axis=-1)
input_patches, input_time_patches, label_patches = create_patches_inference(wind_speed_data, time_steps_label)


# Create label patches with time_steps_label
wind_direction_data = np.expand_dims(wind_direction_data, axis=-1)
input_direction, _, _ = create_patches_inference(wind_direction_data, time_steps_label)

#%%
lat_long_top = np.expand_dims(lat_long_top, axis=-1)

# Renaming for inference
inf_inputs, inf_inputs_time, inf_inputs_direction = input_patches, input_time_patches, input_direction

#%% for the latitude longitude and topology
inf_extra = np.tile(np.expand_dims(np.tile(lat_long_top,reps=inf_inputs.shape[0]),axis=-1),3).transpose(0,3,4,1,2)

# %%
from keras.callbacks import ModelCheckpoint

#%% testing
def transpose_fn(x):
      return tf.transpose(x, perm=[0, 4, 2, 3, 1])
from keras.models import load_model


# Load the saved model
best_model = load_model(f'checkpoints/best_model_20241113_042902_copy.keras',custom_objects={'grad_loss': grad_loss, "transpose_fn": transpose_fn})

# Now you can use `best_model` to make predictions or further evaluations
predictions = best_model.predict([inf_inputs, inf_inputs_time, inf_extra[0], inf_extra[1], inf_extra[2], inf_inputs_direction],)



from scipy.ndimage import zoom
def trilinear_downscale(inputs, target_size):
    """
    Downscales the second dimension (dimension 1) using trilinear interpolation.
    
    Args:
        inputs (np.ndarray): Input array of shape (N, D, H, W, C)
        target_size (int): The target size for the second dimension (D)
    
    Returns:
        np.ndarray: Downscaled array of shape (N, new_D, H, W, C)
    """
    # Compute the zoom factor for downscaling
    zoom_factor = target_size / inputs.shape[1]
    
    # Create a zoom array where the zoom factor is applied to dimension 1
    zoom_factors = [1.0, zoom_factor, 1.0, 1.0, 1.0]
    
    # Apply the zoom function
    downscaled = zoom(inputs, zoom_factors, order=1)  # order=1 for trilinear interpolation
    
    return downscaled
target_size = 5  # New size for the dimension 1

trilinear_interpolation = trilinear_downscale(inf_inputs, target_size)
trilinear_interpolation = trilinear_interpolation[:, [1, 3], :, :, :]


# [15,15] because we removed upper left corner during processing so the wanted point is in this index
point_prediction = predictions[:,:,15,15,0]
# averaging predictions to have more robust results
def compute_average(point_prediction):
    first, second = point_prediction[:,0], point_prediction[:,1]
    rolled_second = np.roll(second, shift=1)
    first = np.append(first, rolled_second[0])
    rolled_second = np.append(rolled_second, rolled_second[0])
    rolled_second[0] = first[0]

    averaged = (first + rolled_second) / 2
    return averaged

point_prediction_averaged = compute_average(point_prediction)
print(point_prediction_averaged.shape)

# create also for the trilinear interpolation to compare
point_trilinear_interpolation = trilinear_interpolation[:,:,15,15,0]
point_trilinear_interpolation_averaged =  compute_average(point_trilinear_interpolation)

### Reconstructing actual wind speed
# Load the normalization parameters from the JSON file
with open('normalization_params.json', 'r') as f:
    normalization_params = json.load(f)

# Extract wind speed min and max from the loaded parameters
wind_speed_min = normalization_params["wind_speed_min"]
wind_speed_max = normalization_params["wind_speed_max"]

# Reverse normalization for all values in point_prediction_averaged
prediction_values = point_prediction_averaged * (wind_speed_max - wind_speed_min) + wind_speed_min
print(f"Prediction values: {prediction_values}")

# Reverse norm for trilinear
trilinear_values = point_trilinear_interpolation_averaged * (wind_speed_max - wind_speed_min) + wind_speed_min
print(f"Prediction values: {trilinear_values}")

# import csv
# # loop to combine the data and write to csv
# def combine_and_save_csv(dataset_points, predicted_points, output_filename='combined_points.csv'):
#     """
#     Combines dataset points and predicted points into a CSV, alternating between them in columns.
#     """
    
#     # Create an empty list to store the alternating points in column pairs
#     combined_points = []

#     # Alternate between dataset_points and predicted_points
#     for i in range(len(dataset_points)):
#         combined_points.append([dataset_points[i]])
#         if i == len(dataset_points) - 1:
#             break
#         combined_points.append([predicted_points[i]])

#     # Write to CSV with two columns: "Dataset Point" and "Predicted Point"
#     with open(output_filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(combined_points)

#     print(f"CSV file with combined points saved as '{output_filename}'.")
# year = 2019

# combine_and_save_csv(wind_speed_data_point, prediction_values, f'combined_points_DNN_{year}_{model_res}.csv')
# combine_and_save_csv(wind_speed_data_point, trilinear_values, f'combined_points_trilinear_{year}.csv')

import csv
from datetime import datetime, timedelta

def combine_and_save_csv(dataset_points, predicted_points, year, interval_minutes=5, output_filename='combined_points.csv'):
    """
    Combines dataset points and predicted points into a CSV with timestamps starting from the beginning of the given year.
    """
    
    # Create an empty list to store the combined points with timestamps
    combined_points = []

    # Set the start date to the beginning of the given year
    start_date = datetime(year, 1, 1, 0, 0)  # Year, Month=1, Day=1, Hour=0, Minute=0
    current_time = start_date

    # Loop through the dataset points and predicted points
    for i in range(len(dataset_points)):
        # Generate the timestamp for the dataset point
        year = current_time.year
        month = current_time.month
        day = current_time.day
        hour = current_time.hour
        minute = current_time.minute
        
        # Add the row for the dataset point with the timestamp
        combined_points.append([
            year, month, day, hour, minute, dataset_points[i]  # N/A for Wind Direction (to be replaced if available)
        ])
        
        # Update the timestamp for the next prediction
        current_time += timedelta(minutes=interval_minutes)
        
        # If there's a predicted point, add that in the next row
        if i < len(predicted_points):
            year = current_time.year
            month = current_time.month
            day = current_time.day
            hour = current_time.hour
            minute = current_time.minute
            
            combined_points.append([
                year, month, day, hour, minute, predicted_points[i]  # N/A for Wind Direction
            ])
        
        # Update the timestamp for the next dataset point
        current_time += timedelta(minutes=interval_minutes)

    # Write to CSV with columns: Year, Month, Day, Hour, Minute, Wind Speed, Wind Direction
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Year", "Month", "Day", "Hour", "Minute", "Wind Speed"])
        # Write data
        writer.writerows(combined_points)

    print(f"CSV file with combined points saved as '{output_filename}'.")

# Year for the data
# year = 2020
interval_minutes = 5
# Call the function to save the CSV
combine_and_save_csv(wind_speed_data_point, prediction_values, year, interval_minutes, f'combined_points_DNN_{year}_{model_res}.csv')
combine_and_save_csv(wind_speed_data_point, trilinear_values, year, interval_minutes, f'combined_points_trilinear_{year}.csv')

