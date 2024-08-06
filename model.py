#%%
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

print(tf.__version__)
#%%
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
                         padding        = 'valid',
                         data_format    = "channels_last")(unet)
    
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

def main():
  model = get_model() # DCN
#   model = get_model(residual=True) # RPN
  model.summary()
  print(model.summary())

#%%
if __name__ == '__main__':
    main()


# %% 
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
wind_speed_data = np.load('tensor.npy')  # Assuming wind speed is in the first channel
wind_speed_data = wind_speed_data[:, :, :, 0]  # Extract wind speed channel

# Normalize the data
wind_speed_data = (wind_speed_data - np.min(wind_speed_data)) / (np.max(wind_speed_data) - np.min(wind_speed_data))
# %%
# Reshape data to fit the model input shape
def reshape_data(data, patch_size=32, time_steps=3):
    num_samples = data.shape[0] - time_steps + 1
    patches = []
    for i in range(num_samples):
        patch = data[i:i+time_steps, :, :]
        patches.append(patch)
    patches = np.array(patches)
    patches = np.expand_dims(patches, axis=-1)  # Add channel dimension
    return patches

# Example reshape
patch_size = 16
time_steps = 3
wind_speed_patches = reshape_data(wind_speed_data, patch_size, time_steps)
# %%
# Split data
train_data, val_data = train_test_split(wind_speed_patches, test_size=0.2, random_state=42)
# %%
# Train the model
epochs = 50
batch_size = 16
model = get_model(PS=batch_size)
history = model.fit([train_data, train_data, train_data, train_data, train_data], 
                    train_data[:, 1, :, :, :],
                    validation_data=([val_data, val_data, val_data, val_data, val_data], val_data[:, 1, :, :, :]),
                    epochs=epochs,
                    batch_size=batch_size)

# %%
