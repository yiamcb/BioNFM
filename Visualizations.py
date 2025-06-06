import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

fd_data = np.mean(extracted_features[:, :, :, 1], axis=(1, 2))  # Average FD across frames and channels

# Separate data based on labels
healthy_fd = fd_data[np.where(all_labels == "Healthy")[0]]
pd_fd = fd_data[np.where(all_labels == "PD")[0]]

# Statistical Comparison
stat, p_value = ttest_ind(healthy_fd, pd_fd)

summary_table = pd.DataFrame({
    "Group": ["Healthy", "PD"],
    "Mean FD": [np.mean(healthy_fd), np.mean(pd_fd)],
    "Std FD": [np.std(healthy_fd), np.std(pd_fd)],
    "p-value": [p_value, p_value]
})

print("Fractal Dimension Summary")
print(summary_table)

x_labels = ["Healthy", "PD"]
x_positions = range(len(x_labels))
mean_values = [np.mean(healthy_fd), np.mean(pd_fd)]
std_values = [np.std(healthy_fd), np.std(pd_fd)]

plt.figure(figsize=(10, 6))

plt.plot(x_positions, mean_values, marker="o", linestyle="-", color="navy", linewidth=2.5, label="Mean FD")

plt.fill_between(x_positions,
                 [m - s for m, s in zip(mean_values, std_values)],
                 [m + s for m, s in zip(mean_values, std_values)],
                 color="lightblue", alpha=0.5, label="Std Deviation")

for i, (mean, std) in enumerate(zip(mean_values, std_values)):
    plt.scatter(x_positions[i], mean, color="darkred", s=100, edgecolors="black", label="Group Points" if i == 0 else "")

plt.xticks(x_positions, x_labels, fontsize=12)
plt.yticks(fontsize=12)
plt.title("Fractal Dimension Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Fractal Dimension", fontsize=12)
plt.xlabel("Group", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(fontsize=12, loc="upper right")
plt.tight_layout()
plt.savefig("fractalDims.pdf", format="pdf", bbox_inches="tight")

# Display the plot
plt.show()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns

entropy_data = np.mean(extracted_features[:, :, :, 4], axis=1)  # Average entropy over frames (shape: samples, n_channels)

# Separate data into Healthy and PD groups
healthy_entropy = entropy_data[np.where(all_labels == "Healthy")[0]]
pd_entropy = entropy_data[np.where(all_labels == "PD")[0]]

# Compute mean and standard deviation across channels
healthy_mean = np.mean(healthy_entropy, axis=0)
healthy_std = np.std(healthy_entropy, axis=0)
pd_mean = np.mean(pd_entropy, axis=0)
pd_std = np.std(pd_entropy, axis=0)

# Statistical comparison for each channel
channel_p_values = [ttest_ind(healthy_entropy[:, ch], pd_entropy[:, ch])[1] for ch in range(entropy_data.shape[1])]

channel_indices = range(entropy_data.shape[1])
comparison_df = pd.DataFrame({
    "Channel": channel_indices,
    "Healthy Mean": healthy_mean,
    "PD Mean": pd_mean,
    "p-value": channel_p_values
})

print("Entropy Feature Comparison by Channel")
print(comparison_df)

plt.figure(figsize=(12, 8))
plt.errorbar(channel_indices, healthy_mean, yerr=healthy_std, fmt="-o", label="Healthy", color="blue", capsize=5, linewidth=2)
plt.errorbar(channel_indices, pd_mean, yerr=pd_std, fmt="-o", label="PD", color="red", capsize=5, linewidth=2)

plt.title("Entropy-Based Features Across Channels", fontsize=16, fontweight="bold")
plt.xlabel("Channel Index", fontsize=14)
plt.ylabel("Entropy Value (Mean ± STD)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("Entropy Features.pdf", format="pdf", bbox_inches="tight")
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import seaborn as sns

pac_data = np.mean(extracted_features[:, :, :, 3], axis=1)  # Average PAC over frames (shape: samples, n_channels)

# Separate data into Healthy and PD groups
healthy_pac = pac_data[np.where(all_labels == "Healthy")[0]]  # Shape: (n_healthy_samples, n_channels)
pd_pac = pac_data[np.where(all_labels == "PD")[0]]  # Shape: (n_pd_samples, n_channels)

healthy_mean = np.mean(healthy_pac, axis=0)
healthy_std = np.std(healthy_pac, axis=0)
pd_mean = np.mean(pd_pac, axis=0)
pd_std = np.std(pd_pac, axis=0)

channel_p_values = [ttest_ind(healthy_pac[:, ch], pd_pac[:, ch])[1] for ch in range(pac_data.shape[1])]

channel_indices = range(pac_data.shape[1])
comparison_df = pd.DataFrame({
    "Channel": channel_indices,
    "Healthy Mean PAC": healthy_mean,
    "PD Mean PAC": pd_mean,
    "p-value": channel_p_values
})

print("PAC Feature Comparison by Channel")
print(comparison_df)

plt.figure(figsize=(12, 8))
plt.errorbar(channel_indices, healthy_mean, yerr=healthy_std, fmt="-o", label="Healthy", color="green", capsize=5, linewidth=2)
plt.errorbar(channel_indices, pd_mean, yerr=pd_std, fmt="-o", label="PD", color="orange", capsize=5, linewidth=2)

plt.title("PAC Feature Across Channels", fontsize=16, fontweight="bold")
plt.xlabel("Channel Index", fontsize=14)
plt.ylabel("PAC Value (Mean ± STD)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("PAC.pdf", format="pdf", bbox_inches="tight")
plt.show()



import numpy as np
import pandas as pd
from scipy.stats import f_oneway

regions = {
    "Frontal": ["F3", "F7", "Fz", "F4", "F8"],
    "Temporal": ["T7", "T8"],
    "Parietal": ["P3", "Pz", "P4"],
    "Occipital": ["O1", "Oz", "O2"]
}

channel_names = ["F3", "F7", "Fz", "F4", "F8", "T7", "T8", "P3", "Pz", "P4", "O1", "Oz", "O2"]
channel_indices = {name: idx for idx, name in enumerate(channel_names)}

region_indices = {region: [channel_indices[ch] for ch in channels] for region, channels in regions.items()}

regional_features = {}
for region, indices in region_indices.items():
    regional_features[region] = [
        np.mean(extracted_features[:, :, indices, i], axis=(1, 2)) for i in range(5)
    ]  # Average over specified channels and frames

healthy_indices = np.where(all_labels == "Healthy")[0]
pd_indices = np.where(all_labels == "PD")[0]

anova_results = []
for region, features in regional_features.items():
    for i, feature_data in enumerate(features):
        healthy_data = feature_data[healthy_indices]
        pd_data = feature_data[pd_indices]

        # Perform ANOVA
        stat, p_value = f_oneway(healthy_data, pd_data)
        anova_results.append({
            "Region": region,
            "Feature": f"Feature {i + 1}",
            "Healthy Mean": np.mean(healthy_data),
            "PD Mean": np.mean(pd_data),
            "F-Statistic": stat,
            "p-value": p_value
        })

anova_table = pd.DataFrame(anova_results)

print("ANOVA Results Table Grouped by Region and Feature")
print(anova_table)


