import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['Accuracy', 'Local Relevance', 'Completeness', 'Usefulness']
custom = [4.8, 4.6, 3.966666667, 4.233333333]
baseline = [4.033333333, 4.366666667, 4.1, 4.1]

# X positions for bars
x = np.arange(len(metrics))
width = 0.35  # Width of the bars

# Create the figure and axes
fig, ax = plt.subplots(figsize=(9, 5))

# Bars for each model
bars1 = ax.bar(x - width/2, custom, width, label='Custom Model', color='orange')
bars2 = ax.bar(x + width/2, baseline, width, label='Baseline Model', color='skyblue')

# Add labels and title
ax.set_ylabel('Average Score (0–5)', fontsize=12)
ax.set_xlabel('Evaluation Metric', fontsize=12)
#ax.set_title('Comparison of Custom vs Baseline Model Evaluation Scores', fontsize=15, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Annotate bars with values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=10)

# Set y-axis limit for cleaner spacing
ax.set_ylim(0, 6)

# Improve layout
plt.tight_layout()
plt.show()
