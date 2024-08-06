# %%
import numpy as np
import matplotlib.pyplot as plt

# Load your wind speed data
wind_speed_data = np.load('tensor.npy')  # Assuming wind speed is in the first channel
wind_speed_data = wind_speed_data[:, :, :, 0]  # Extract wind speed channel

# Temporal analysis
def temporal_analysis(data):
    time_series = data.mean(axis=(1, 2))  # Average over spatial dimensions
    plt.plot(time_series)
    plt.title("Wind Speed Time Series")
    plt.xlabel("Time")
    plt.ylabel("Wind Speed")
    plt.show()
    print("Temporal Standard Deviation:", np.std(time_series))

# Spatial analysis
def spatial_analysis(data, time_index):
    plt.imshow(data[time_index, :, :], cmap='viridis')
    plt.title(f"Wind Speed Spatial Plot at Time Index {time_index}")
    plt.colorbar(label='Wind Speed')
    plt.show()

# Example usage
temporal_analysis(wind_speed_data)
spatial_analysis(wind_speed_data, time_index=10)

# %%
