import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

left_channel_indices = [0, 1, 2, 3, 4, 5]  # F3, F7, C3, T7, P3, O1
right_channel_indices = [6, 7, 8, 9, 10, 11]  # F4, F8, C4, T8, P4, O2

# Extract Beta Band Feature
beta_band_data = extracted_features[:, :, :, 0]

# Compute inter-hemispheric coherence for each sample
n_samples = beta_band_data.shape[0]
inter_hemispheric_coherence = []

for i in range(n_samples):
    left_mean = np.mean(beta_band_data[i, :, left_channel_indices], axis=0)  # Mean across frames for left channels
    right_mean = np.mean(beta_band_data[i, :, right_channel_indices], axis=0)  # Mean across frames for right channels
    coherence = np.corrcoef(left_mean, right_mean)[0, 1]  # Correlation coefficient
    inter_hemispheric_coherence.append(coherence)

inter_hemispheric_coherence = np.array(inter_hemispheric_coherence)


healthy_coherence = inter_hemispheric_coherence[np.where(all_labels == "Healthy")[0]]
pd_coherence = inter_hemispheric_coherence[np.where(all_labels == "PD")[0]]

# Statistical Comparison
stat, p_value = ttest_ind(healthy_coherence, pd_coherence)

summary_table = pd.DataFrame({
    "Group": ["Healthy", "PD"],
    "Mean Coherence": [np.mean(healthy_coherence), np.mean(pd_coherence)],
    "Std Coherence": [np.std(healthy_coherence), np.std(pd_coherence)],
    "p-value": [p_value, p_value]
})

print("Beta Band Inter-hemispheric Coherence Summary")
print(summary_table)

import seaborn as sns

data_to_plot = pd.DataFrame({
    "Coherence": np.concatenate([healthy_coherence, pd_coherence]),
    "Group": ["Healthy"] * len(healthy_coherence) + ["PD"] * len(pd_coherence)
})

plt.figure(figsize=(8, 6))
sns.stripplot(x="Group", y="Coherence", data=data_to_plot, jitter=True, palette="muted", alpha=0.7)
sns.pointplot(x="Group", y="Coherence", data=data_to_plot, ci="sd", join=False, color="black", markers="D", scale=1.2)
plt.title("Beta Band Inter-hemispheric Coherence")
plt.ylabel("Coherence")
plt.savefig("Beta1.pdf", format="pdf", bbox_inches="tight")
plt.show()


plt.figure(figsize=(8, 6))
sns.kdeplot(healthy_coherence, label="Healthy", fill=True, color="blue", alpha=0.5)
sns.kdeplot(pd_coherence, label="PD", fill=True, color="red", alpha=0.5)
plt.title("Beta Band Inter-hemispheric Coherence Distribution")
plt.xlabel("Coherence")
plt.legend()
plt.savefig("Beta2.pdf", format="pdf", bbox_inches="tight")
plt.show()