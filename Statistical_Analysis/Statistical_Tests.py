import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind

# Translation dictionary for transportation modes
translation_dict = {
    'Voiture / Moto': 'Car / Motorbike', 
    'Transport collectif': 'Public Transport',
    'Vélo': 'Bicycle',
    'À pied': 'Walking',
    'À pied, Transport collectif': 'Walking + Public Transport',
    'À pied, Transport collectif, Vélo': 'Walking + Public Transport + Bicycle',
    'À pied, Voiture / Moto, Vélo': 'Walking + Car / Motorbike + Bicycle',
    'À pied, Vélo': 'Walking + Bicycle',
    'Transport collectif, Voiture / Moto': 'Public Transport + Car / Motorbike'
}

# Load the dataset
file_path = 'first_10000_rows.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Drop unnecessary columns
data_cleaned = data.drop(columns=['altitude', 'v_accuracy', 'h_accuracy'])

# Translate mode column to English
data_cleaned['mode'] = data_cleaned['mode'].replace(translation_dict)

# Convert timestamps and calculate trip duration
data_cleaned['starttime'] = pd.to_datetime(data_cleaned['starttime'])
data_cleaned['endtime'] = pd.to_datetime(data_cleaned['endtime'])
data_cleaned['timestamp'] = pd.to_datetime(data_cleaned['timestamp'])

# Add derived column for trip duration in minutes
data_cleaned['trip_duration'] = (data_cleaned['endtime'] - data_cleaned['starttime']).dt.total_seconds() / 60

# Drop rows with missing or invalid values (if any)
data_cleaned = data_cleaned.dropna()

# Perform ANOVA: Test if average speed differs by transportation mode
modes = data_cleaned['mode'].unique()
speed_groups = [data_cleaned[data_cleaned['mode'] == mode]['speed'] for mode in modes]
anova_result = f_oneway(*speed_groups)

# Save ANOVA results to a text file
anova_results_file = "anova_results.txt"
with open(anova_results_file, "w") as file:
    file.write("### ANOVA Results (Speed by Transportation Mode) ###\n")
    file.write(f"F-Statistic: {anova_result.statistic:.3f}\n")
    file.write(f"P-Value: {anova_result.pvalue:.3e}\n")
    if anova_result.pvalue < 0.05:
        file.write("Result: Significant differences in average speeds across modes (p < 0.05).\n")
    else:
        file.write("Result: No significant differences in average speeds across modes (p >= 0.05).\n")
print(f"ANOVA results saved to {anova_results_file}")

# Perform Pairwise T-tests
t_test_summary = []
pairwise_p_values = np.zeros((len(modes), len(modes)))

for i, mode_1 in enumerate(modes):
    for j, mode_2 in enumerate(modes):
        if i != j:
            speed_mode_1 = data_cleaned[data_cleaned['mode'] == mode_1]['speed']
            speed_mode_2 = data_cleaned[data_cleaned['mode'] == mode_2]['speed']
            ttest_result = ttest_ind(speed_mode_1, speed_mode_2)
            t_test_summary.append({
                "Mode 1": mode_1,
                "Mode 2": mode_2,
                "T-Statistic": ttest_result.statistic,
                "P-Value": ttest_result.pvalue
            })
            pairwise_p_values[i, j] = ttest_result.pvalue

# Save T-test summary to a CSV file
t_test_results_file = "t_test_results.csv"
t_test_summary_df = pd.DataFrame(t_test_summary)
t_test_summary_df.to_csv(t_test_results_file, index=False)
print(f"T-test summary saved to {t_test_results_file}")

# Create a DataFrame for the heatmap
p_value_df = pd.DataFrame(pairwise_p_values, index=modes, columns=modes)

# Visualize the results as a heatmap and save the plot
plt.figure(figsize=(12, 8))
sns.heatmap(p_value_df, annot=True, fmt=".2e", cmap="coolwarm", cbar_kws={'label': 'P-Value'})
plt.title("Pairwise T-Test P-Values for Transportation Modes")
plt.xlabel("Mode 2")
plt.ylabel("Mode 1")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Save the heatmap plot
heatmap_file = 'pairwise_ttest_heatmap.png'
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
plt.show()
print(f"Heatmap saved as {heatmap_file}")
