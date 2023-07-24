import numpy as np
import matplotlib.pyplot as plt


# Define a function to calculate the fractal dimension of a curve using the box-counting method
def fractal_dimension(x, y, box_sizes):
    box_counts = []
    for box_size in box_sizes:
        x_bins = np.arange(np.min(x), np.max(x) + box_size, box_size)
        y_bins = np.arange(np.min(y), np.max(y) + box_size, box_size)
        counts, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        box_counts.append(np.sum(counts > 0))
    box_counts = np.array(box_counts)
    box_sizes = np.array(box_sizes)
    log_counts = np.log(box_counts)
    log_sizes = np.log(box_sizes)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return slope


x = np.arange(0, 10, 0.1)
box_sizes = np.arange(0.1, 1, 0.1)
y = []

slopes = []
for i in range(100):
    y.append(np.sin(x) + np.random.normal(0, 0.1*i, x.shape))
    slope = fractal_dimension(x, y[-1], box_sizes)
    slopes.append(slope)


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for i in range(2):
    for j in range(2):
        ax[i, j].plot(x, y[i*2+j])
        ax[i, j].set_title(f'slope = {slopes[i*2+j]:.2f}')

plt.tight_layout()
plt.show()
