from file_names import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(file_name_2, parse_dates=['time'])
# df['speed'] = np.hypot(df['velocity_north'], df['velocity_east'])
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

plt.savefig('figures/number_of_segments_per_drifter_stacked_death_code1.png', dpi=300)
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
plt.savefig('figures/2d_histogram_beaching_per_drifter_death_code1.png', dpi=300)
plt.show()

#%%
n_false_ground = table_beaching_per_drifter[False][table_beaching_per_drifter[True] == 1]
n_false_no_ground = table_beaching_per_drifter[False][table_beaching_per_drifter[True] != 1]

# plot both in histogram
max_value = np.max([np.max(n_false_ground), np.max(n_false_no_ground)])

dbin = 40

bin_edges = np.arange(0, max_value + dbin, dbin)  # Ensure the last bin edge is included

# Count the number of drifters in each bin
counts_ground, _ = np.histogram(n_false_ground, bins=bin_edges)
counts_no_ground, _ = np.histogram(n_false_no_ground, bins=bin_edges)

counts_ground_state = np.zeros(len(bin_edges) - 1)
counts_no_ground_state = np.zeros(len(bin_edges) - 1)
for i, (lower_limit, upper_limit) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    counts_ground_state[i] = np.sum(n_false_ground[(n_false_ground >= lower_limit) & (n_false_ground < upper_limit)])
    counts_no_ground_state[i] = np.sum(n_false_no_ground[(n_false_no_ground >= lower_limit) & (n_false_no_ground < upper_limit)])


# The x locations for the groups
ind = np.arange(len(counts_ground))
# The width of the bars
width = 0.5

fig, ax = plt.subplots(figsize=(10, 5))

rects1 = ax.bar(ind+0.25, counts_ground, width, label='grounding', color='blue')
rects2 = ax.bar(ind+0.75, counts_no_ground, width, label='non-grounding', color='red')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Number of non-grounding drifter states per drifter')
ax.set_ylabel('Number of drifters')
ax.set_xticks(ind)
ax.set_xticklabels(range(0, max_value, dbin))
ax.legend(loc='upper left')

ax2 = ax.twinx()
ax2.plot(ind+0.25, counts_ground_state, 'o:', color=(0.3, 0.3, 1), markersize=4, label='grounding')
ax2.plot(ind+0.75, counts_no_ground_state, 's:', color=(1, 0.3, 0.3), markersize=4, label='non-grounding')
ax2.set_ylabel('Number of non-grounding drifter states')
ax2.legend(loc='upper right')

#log y scale
# ax.set_yscale('log', base=4)

fig.tight_layout()
plt.savefig('figures/no_grounding_per_drifter_death_code1.png', dpi=300)
plt.show()


