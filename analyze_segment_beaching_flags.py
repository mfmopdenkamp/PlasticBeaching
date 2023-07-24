from file_names import file_name_4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(file_name_4, parse_dates=['time_start', 'time_end'], index_col='ID')
df['speed'] = np.hypot(df['velocity_north'], df['velocity_east'])
table_beaching_per_drifter = df.groupby('drifter_id').beaching_flag.value_counts().unstack().fillna(0).astype(int)
# sort the table descending from most to least false beaching flags
table_beaching_per_drifter.sort_values(by=False, ascending=False, inplace=True)

#%% depict how represented each drifter is in the dataset for a true and false beaching flag by a stacked bar plot
table_beaching_per_drifter.plot(kind='bar', stacked=True, figsize=(15, 5), title='Number of segments per drifter',
                                align='edge', width=1)
plt.legend()
plt.xlabel('Drifters')
plt.ylabel('Number of trajectory segments with true and false beaching flags')

ax = plt.gca()

# Clear the major ticks and tick labels
ax.set_xticks([])
ax.set_xticklabels([])

plt.savefig('figures/number_of_segments_per_drifter_stacked.png', dpi=300)
plt.show()

print(f'Number of unique drifters: {len(df.drifter_id.unique())}')

#%% plot a 1x1 2d histogram plot of number of true beaching flags vs number of false beaching flags per drifter

# 2d histogram
# Assuming 'x' and 'y' are the data arrays for the x and y coordinates
x = table_beaching_per_drifter[True]
y = table_beaching_per_drifter[False]

# Determine the extent of the grid
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

# Calculate the number of bins along each axis based on cell size 1x1
num_bins_x = int(np.ceil(x_max - x_min))
num_bins_y = int(np.ceil(y_max - y_min))

# Create a figure and axis
fig, ax = plt.subplots(figsize=(int(np.ceil(num_bins_y/4)), int(np.ceil(num_bins_x/4))))

# Plot the 2D histogram with specified extent and aspect ratio
h, xedges, yedges, _ = ax.hist2d(x, y, bins=[num_bins_x, num_bins_y], cmap='Blues')

# Iterate over the bins and annotate each cell with its count
for i in range(len(xedges) - 1):
    for j in range(len(yedges) - 1):

        count = int(h[i, j])  # Get the count for the current cell
        x_pos = xedges[i] + 0.5 * (xedges[i + 1] - xedges[i])  # Calculate x position for annotation
        y_pos = yedges[j] + 0.5 * (yedges[j + 1] - yedges[j])  # Calculate y position for annotation
        ax.annotate(str(count), xy=(x_pos, y_pos), ha='center', va='center',
                    color=('black' if 200 > count > 0 else ('white' if count > 200 else (0.7, 0.7, 0.7))))


# Set the labels and title
plt.xlabel(r'Number of segments $\bf{with}$ grounding')
plt.ylabel(r'Number of segments $\bf{without}$ grounding')
ax.set_title('Distribution of drifters by amount of segments with and without grounding')

# Remove original ticks and ticklabels
ax.tick_params(axis='both', which='both', length=0)

# Shift the ticks and ticklabels by 0.5
ax.set_xticks(np.arange(0, num_bins_x , 1) + 0.5, minor=True)
ax.set_yticks(np.arange(0, num_bins_y, 1) + 0.6, minor=True)
ax.set_xticklabels(np.arange(0, num_bins_x, 1), minor=True)
ax.set_yticklabels(np.arange(0, num_bins_y, 1), minor=True)

# Clear the major ticks and tick labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# Display the plot
plt.savefig('figures/2d_histogram_beaching_per_drifter.png', dpi=300)
plt.show()

