import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Metrics for each model
metrics = {'Model': ['Majority', '1-Split', 'Decision Tree', 'Random Forest'],
           'Accuracy': [0.9087378640776699, 0.9009708737864077, 0.9106796116504854, 0.9184466019417475],
           'Precision': [0.0, 0.43333333333333335, 0.5555555555555556, 0.6190476190476191],
           'Recall': [0.0, 0.2765957446808511, 0.10638297872340426, 0.2765957446808511],
           'F1': [0.0, 0.33766233766233766, 0.17857142857142855, 0.38235294117647056]}

# Create dataframe
df = pd.DataFrame(metrics)
# Set index
df.set_index('Model', inplace=True)

x=np.arange(len(df.index))  # the label locations
width = 1/len(df.columns)-0.05  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

# Plot bar chart for Accuracy
ax.bar(x-1.5*width, df['Accuracy'], color='blue', width=width, label='Accuracy')

# Set y-axis limits
ax.set_ylim([0.9, 0.92])
# Set y-axis label
ax.set_ylabel('Accuracy')
# Create twin axis
ax2 = ax.twinx()
# Plot other metrics (drop Accuracy column before plotting)
ax2.bar(x-0.5*width, df['Precision'], color='red', width=width, label='Precision')
ax2.bar(x+0.5*width, df['Recall'], color='green', width=width, label='Recall')
ax2.bar(x+1.5*width, df['F1'], color='orange', width=width, label='F1')

ax.set_xticks(x)
ax.set_xticklabels(df.index)

ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.savefig('figures/model_results_all.png', dpi=300, bbox_inches='tight')
plt.show()

