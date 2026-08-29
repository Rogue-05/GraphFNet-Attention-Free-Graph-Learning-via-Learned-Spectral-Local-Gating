import matplotlib.pyplot as plt
import numpy as np

k_values = [4, 8, 16, 32, 64]
test_ap   = [0.6289, 0.6270, 0.6161, 0.5989, 0.6076]

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(k_values, test_ap, 'o-', color='#c0392b', linewidth=2.5,
        markersize=8, zorder=3)

# Annotate each point
for k, ap in zip(k_values, test_ap):
    ax.annotate(f'{ap:.4f}', (k, ap),
                textcoords='offset points',
                xytext=(0, 10), ha='center',
                fontsize=9, color='#c0392b')

# Baseline reference line (k=8, your reported 3-seed result)
ax.axhline(y=0.6244, color='gray', linestyle='--', linewidth=1.2,
           label='3-seed baseline (k=8): 0.6244')

ax.set_xscale('log', base=2)
ax.set_xticks(k_values)
ax.set_xticklabels([str(k) for k in k_values])
ax.set_xlabel('Number of LapPE Eigenvectors (k)', fontsize=12)
ax.set_ylabel('Test AP', fontsize=12)
ax.set_title('k Sensitivity: LapPE Eigenvector Count vs Performance\n(Seed 0)', fontsize=12)
ax.set_ylim(0.58, 0.645)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('k_sensitivity.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved to k_sensitivity.png')