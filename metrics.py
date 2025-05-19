import os
import re
import pandas as pd

# Directory containing log files
log_dir = 'logs'

# Regular expression to extract metric data
pattern = re.compile(r"For top(\d+), metric (\w+) = ([\deE\.\+-]+)")

# Dictionary to hold results per file
all_data = []

# Process each log file
for filename in sorted(os.listdir(log_dir)):
    if not filename.endswith(".log"):
        continue

    file_path = os.path.join(log_dir, filename)
    metrics = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                top_k, metric, value = match.groups()
                key = f"{metric}@{top_k}"
                metrics[key] = float(value)
    
    # Add filename as identifier
    metrics["filename"] = filename
    all_data.append(metrics)

# Create DataFrame
df = pd.DataFrame(all_data)
df = df.set_index("filename")

# Save to Excel
output_excel = "log_metrics_summary.xlsx"
df.to_excel(output_excel)

print(f"Metrics successfully saved to {output_excel}")
