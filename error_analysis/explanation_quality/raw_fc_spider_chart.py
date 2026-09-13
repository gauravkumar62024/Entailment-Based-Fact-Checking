import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Data for the models
labels = ['Informativeness', 'Accuracy', 'Readability', 'Objectivity', 'Logicality']
num_vars = len(labels)

# Average scores for each model (averaged across all graphs)
avg_gemma   = [3.2682, 2.7072, 4.0644, 3.7184, 3.2672]
avg_falcon  = [2.861,  2.939,  3.3296, 3.3992, 2.955]
avg_qwen    = [2.7098, 2.6932, 3.6852, 3.1332, 3.1904]
avg_mistral = [3.4888, 3.2668, 3.7408, 3.7938, 3.8268]
avg_llama   = [4.0164, 3.2988, 3.8994, 3.5166, 3.6028]

# Combine data into a list
data = [avg_gemma, avg_falcon, avg_qwen, avg_mistral, avg_llama]
model_names = ['Gemma', 'Falcon', 'Qwen', 'Mistral', 'LLaMA']

# Vibrant colors (updated)
colors = [
    '#D62728',  # Gemma  - dark vibrant red
    '#8B4513',  # Falcon - vibrant blue
    '#2CA02C',  # Qwen   - vibrant green
    '#FF7F0E',  # Mistral- vibrant orange
    '#9467BD'   # LLaMA  - vibrant purple
]

# Radar scale range (2 to 5)
r_min, r_max = 2, 5

# Compute angle for each category
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Complete the loop

# Initialize the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw axis labels
plt.xticks(angles[:-1], labels, color="black", size=10, weight='bold')

# Remove default circular grid
ax.grid(False)

# Draw pentagon boundaries manually (2 to 5)
for r in range(r_min, r_max + 1):
    pentagon = [r] * num_vars
    pentagon = np.append(pentagon, pentagon[0])
    ax.plot(angles, pentagon, color='#D3D3D3', linestyle='--', linewidth=2)

# Draw dashed spokes (radial lines) from 2 to 5
for angle in angles[:-1]:
    ax.plot([angle, angle], [r_min, r_max], color='#D3D3D3', linestyle='--', linewidth=1)

# Draw ylabels (scale) from 2 to 5
ax.set_rlabel_position(0)
plt.yticks(
    list(range(r_min, r_max + 1)),
    [str(x) for x in range(r_min, r_max + 1)],
    color="black",
    size=10,
    weight='bold'
)
plt.ylim(r_min, r_max)

# Plot each model's data
for i, model_data in enumerate(data):
    values = model_data + model_data[:1]  # Complete the loop
    ax.plot(
        angles, values,
        linewidth=2, linestyle='solid',
        label=model_names[i],
        color=colors[i]
    )
    ax.fill(angles, values, color=colors[i], alpha=0.25)

    # Add bold points where the line touches the axes
    for j in range(num_vars):
        ax.plot(angles[j], model_data[j], marker='o', markersize=10, color=colors[i])

# Add a legend (stronger visibility)
leg = plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh.set_linewidth(3)

# Remove the outer circle (spine)
ax.spines['polar'].set_visible(False)

# Save the plot
plt.savefig('radar_chart_average_data.png')
plt.savefig('radar_chart_average_data_rawfc.pdf', format='pdf', bbox_inches='tight')

