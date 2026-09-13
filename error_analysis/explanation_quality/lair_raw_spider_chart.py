import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Labels for evaluation criteria
labels = ['Informativeness', 'Accuracy', 'Readability', 'Objectivity', 'Logicality']
num_vars = len(labels)

# Average scores for each model
avg_gemma   = [3.5160, 2.8962, 4.0408, 3.7312, 3.4232]
avg_falcon  = [4.0184, 3.7950, 3.3442, 4.0016, 3.9800]
avg_qwen    = [2.9748, 2.7846, 3.8970, 3.6278, 3.4678]
avg_mistral = [4.1598, 3.4638, 3.9292, 4.0968, 4.1078]
avg_llama   = [4.4588, 3.6418, 4.2920, 4.1418, 4.2156]

# Combine into dataset
data = [avg_gemma, avg_falcon, avg_qwen, avg_mistral, avg_llama]
model_names = ['Gemma', 'Falcon', 'Qwen', 'Mistral', 'LLaMA']

# Vibrant colors (updated)
colors = [
    '#D62728',  # Gemma  - dark vibrant red
    '#8B4513',  # Falcon - brown
    '#2CA02C',  # Qwen   - vibrant green
    '#FF7F0E',  # Mistral- vibrant orange
    '#9467BD'   # LLaMA  - vibrant purple
]

# Range 2 to 5
r_min, r_max = 2, 5

# Calculate radar chart angles
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

# Initialize radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Add category labels
plt.xticks(angles[:-1], labels, color="black", size=10, weight='bold')

# Remove default grid
ax.grid(False)

# Draw pentagon boundaries (2..5) with lighter gray
for r in range(r_min, r_max + 1):
    pentagon = [r] * num_vars
    pentagon = np.append(pentagon, pentagon[0])
    ax.plot(angles, pentagon, color='#D3D3D3', linestyle='--', linewidth=2)

# Draw radial lines (2..5) with lighter gray
for angle in angles[:-1]:
    ax.plot([angle, angle], [r_min, r_max], color='#D3D3D3', linestyle='--', linewidth=1)

# Add radial axis labels (2..5)
ax.set_rlabel_position(0)
plt.yticks([2, 3, 4, 5], ["2", "3", "4", "5"], color="black", size=10, weight='bold')
plt.ylim(r_min, r_max)

# Plot each model’s radar line
for i, model_data in enumerate(data):
    values = model_data + model_data[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_names[i], color=colors[i])
    ax.fill(angles, values, color=colors[i], alpha=0.25)
    for j in range(num_vars):
        ax.plot(angles[j], model_data[j], marker='o', markersize=10, color=colors[i])

# Add legend (stronger)
leg = plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh.set_linewidth(3)

# Hide outer circular spine
ax.spines['polar'].set_visible(False)

# Save figure
plt.savefig('radar_chart_model_avg_scores_lair_raw_average_2_to_5.png')
plt.savefig('radar_chart_model_avg_scores_lair_raw_average_2_to_5.pdf', format='pdf', bbox_inches='tight')

