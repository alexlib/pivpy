"""
Example script demonstrating the autocorrelation_plot function in pivpy

This example shows how to use the autocorrelation function to analyze
temporal correlations in PIV data for different variables like u, v,
vorticity (w), or other scalar quantities.
"""

from pivpy import io, graphics, pivpy
import pathlib
import importlib.resources
import matplotlib.pyplot as plt

# Get the path to example data files
try:
    from importlib.resources import files
    path = files('pivpy') / 'data'
except ImportError:
    from importlib.resources import path as resource_path
    with resource_path('pivpy', 'data') as data_path:
        path = data_path

path = pathlib.Path(path)
filename = path / 'Insight' / 'Run000001.T000.D000.P000.H001.L.vec'

# Load PIV data
data = io.load_vec(filename)

# Example 1: Plot autocorrelation of u velocity component using graphics module
# By default, spatial_average=True computes spatial average first for proper temporal analysis
print("Example 1: Autocorrelation of u-component (spatial average)")
ax = graphics.autocorrelation_plot(data, variable='u', spatial_average=True)
plt.show()

# Example 2: Plot autocorrelation with all dimensions flattened (original gist behavior)
print("\nExample 2: Autocorrelation of u-component (all dimensions flattened)")
ax = graphics.autocorrelation_plot(data, variable='u', spatial_average=False)
plt.show()

# Example 3: Plot autocorrelation of v velocity component using accessor method
print("\nExample 3: Autocorrelation of v-component using accessor")
ax = data.piv.autocorrelation_plot(variable='v')
plt.show()

# Example 4: Plot autocorrelation of vorticity (w)
print("\nExample 4: Autocorrelation of vorticity")
data_vort = data.piv.vec2scal('curl')
ax = graphics.autocorrelation_plot(data_vort, variable='w')
plt.show()

# Example 5: Create a multi-panel plot comparing different variables
print("\nExample 5: Multi-panel autocorrelation comparison")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# u-component (spatial average)
plt.sca(axes[0, 0])
graphics.autocorrelation_plot(data, variable='u', spatial_average=True)

# v-component (spatial average)
plt.sca(axes[0, 1])
graphics.autocorrelation_plot(data, variable='v', spatial_average=True)

# u-component (flattened)
plt.sca(axes[1, 0])
graphics.autocorrelation_plot(data, variable='u', spatial_average=False)

# vorticity
data_vort = data.piv.vec2scal('curl')
plt.sca(axes[1, 1])
graphics.autocorrelation_plot(data_vort, variable='w')

plt.tight_layout()
plt.show()

print("\nAutocorrelation plots demonstrate how the signal correlates with itself")
print("at different time lags. A slow decay indicates strong temporal correlations,")
print("while a fast decay suggests the signal varies rapidly in time.")
print("\nWith spatial_average=True (default), the function computes the spatial")
print("average over x and y dimensions first, providing proper temporal autocorrelation.")
print("With spatial_average=False, all dimensions are flattened, which may mix")
print("spatial and temporal variations (as shown in the original gist).")

