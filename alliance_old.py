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

def uNet(input, time, lat, lon, height, kernel = [5, 3, 3], nodes = [72, 144, 288, 576]):
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
    mergetime = Concatenate(axis=4)([conv1, lat, lon, height])
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

    # Load U-Net
    unet        = uNet(main_input, time, lat, lon, height, nodes = nodes)

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
    model       = Model(inputs          = [main_input, time, lat, lon, height],
                        outputs         = temp_out)

    # compile with defined loss and optimizer
    model.compile(loss      = loss,
                  optimizer = optimizer,
                  metrics   = ['mse', 'mae', 'mape'])

    return model


# %%
# Load data
wind_speed_data = np.load('tensor.npy')  # Assuming wind speed is in the first channel
print(wind_speed_data.shape)
wind_direction_data = wind_speed_data[:, :, :, 1]  # Extract wind direction channel
wind_speed_data = wind_speed_data[:, :, :, 0]  # Extract wind speed channel


# Normalize the data
wind_speed_data = (wind_speed_data - np.min(wind_speed_data)) / (np.max(wind_speed_data) - np.min(wind_speed_data))
wind_direction_data = (wind_direction_data - np.min(wind_direction_data)) / (np.max(wind_direction_data) - np.min(wind_direction_data))

print(wind_speed_data.shape, wind_direction_data.shape)



# Reshape data to fit the model input shape
def reshape_data(data, target_shape=(32, 32)):
    # Reshape data to the target shape
    data = data[..., np.newaxis]
    resized_data = tf.image.resize(data, target_shape, method='bilinear')
    data = resized_data.numpy()
    return data

def create_patches(data, time_steps):
    num_samples = data.shape[0] - time_steps + 1
    training_patches = []
    patches = []

    

    for i in range(num_samples):
        patch = data[i:i+time_steps]
        patches.append(patch[1::2])
        training_patches.append(patch[::2])
    training_timestep = time_steps//2+1
    relative_time = np.zeros((time_steps//2+1, data.shape[1]//8, data.shape[2]//8, 1))
    # relative_time = np.zeros((3, 4, 4, 1))

    # Assign specific values to each slice along the first axis
    # relative_time[0, :, :, 0] = -1
    # relative_time[1, :, :, 0] = 0
    # relative_time[2, :, :, 0] = 1
    for t in range(training_timestep):
        relative_time[t, :, :, 0] = t*2
    relative_time_patches = np.tile(relative_time, (num_samples, 1, 1, 1, 1))
    training_patches = np.array(training_patches)
    patches = np.array(patches)

    return training_patches, relative_time_patches, patches


# Parameters
patch_size = 32
time_steps_label = 5

# Reshape data
reshaped_data = reshape_data(wind_speed_data, target_shape=(32, 32))

# Create label patches with time_steps_label
input_patches, input_time_patches, label_patches = create_patches(reshaped_data, time_steps_label)


# Reshape direction data
reshaped_direction = reshape_data(wind_direction_data, target_shape=(32, 32))

# Create label patches with time_steps_label
input_direction, _, _ = create_patches(reshaped_direction, time_steps_label)



# input_patches, input_time_patches = input_patches[:label_patches.shape[0],:,:,:,:], input_time_patches[:label_patches.shape[0],:,:,:,:]

print("Input patches shape:", input_patches.shape)
print("Input time patches shape:", input_time_patches.shape)
print("Label patches shape:", label_patches.shape)


# %% get test size
test_size = input_patches.shape[0]//12
test_inputs = input_patches[-test_size:,:,:,:,:]
test_direction = input_direction[-test_size:,:,:,:,:]

test_inputs_time = input_time_patches[-test_size:,:,:,:,:]

test_labels = label_patches[-test_size:,:,:,:,:]

# get one month
one_month_size = input_patches.shape[0]//12 * 2
input_patches = input_patches[:one_month_size]
input_time_patches = input_time_patches[:one_month_size]
label_patches = label_patches[:one_month_size]
input_direction = input_direction[:one_month_size]

# %%
# Define the size of the training set
train_size = int(0.85 * len(input_patches))  # 80% for training

# Create indices
indices = np.arange(len(input_patches))
np.random.seed(42)  # Set seed for reproducibility
np.random.shuffle(indices)

# Split indices into training and validation
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Use the indices to split the data
train_inputs = input_patches[train_indices]
val_inputs = input_patches[val_indices]

train_inputs_time = input_time_patches[train_indices]
val_inputs_time = input_time_patches[val_indices]

train_labels = label_patches[train_indices]
val_labels = label_patches[val_indices]

train_direction = input_direction[train_indices]
val_direction = input_direction[val_indices]

# %%
from keras.callbacks import ModelCheckpoint

# Define the model
epochs = 100
patch_size = 32
model = get_model(PS=patch_size)

batch_size = 32
# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='best_model_topology.keras',  # Filepath where the model will be saved
    monitor='val_loss',        # Monitor the validation loss
    save_best_only=True,       # Save only the best model
    save_weights_only=False,   # Save the whole model, not just weights
    mode='min',                # Mode 'min' means it will save the model with the minimum validation loss
    verbose=1                  # Verbosity mode
)

# Train the model with validation and checkpointing
history = model.fit(
    [train_inputs, train_inputs_time, train_inputs, train_inputs, train_direction],
    train_labels,
    validation_data=([val_inputs, val_inputs_time, val_inputs, val_inputs, val_direction], val_labels),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint_callback],
    verbose=1
)



#%%
def transpose_fn(x):
      return tf.transpose(x, perm=[0, 4, 2, 3, 1])
from keras.models import load_model

# Load the saved model
best_model = load_model('best_model_topology.keras',custom_objects={'grad_loss': grad_loss, "transpose_fn": transpose_fn})

# Now you can use `best_model` to make predictions or further evaluations
predictions = best_model.predict([test_inputs,test_inputs_time,test_inputs,test_inputs,test_direction])

# mean interpolation
interpolation = [(test_inputs[:,1:2,:,:,:] + test_inputs[:,0:1,:,:,:]) / 2, (test_inputs[:,2:3,:,:,:] + test_inputs[:,1:2,:,:,:]) / 2]
interpolation_result = np.concatenate(interpolation, axis=1).shape

print(f"DCN error{np.mean(np.abs((test_labels - predictions)/ test_labels))}")
print(f"mean interpolation error{np.mean(np.abs((test_labels - interpolation) / test_labels))}")



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
target_size = 2  # New size for the dimension 1

downscaled_interpolation = trilinear_downscale(test_inputs, target_size)

mse_prediction = np.mean((test_labels - predictions) ** 2)
mse_baseline = np.mean((test_labels - interpolation) ** 2)
mse_interpolation = np.mean((test_labels - downscaled_interpolation) ** 2)
print(f"DCN error{mse_prediction}")
print(f"mean interpolation error{mse_baseline}")
print(f"trilinear interpolation error{mse_interpolation}")
