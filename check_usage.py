import subprocess
import json

import pandas as pd

kube_output = subprocess.run(
    ["kubectl", "get", "jobs", "-n", "research", "-o", "json"],
    check=True,
    capture_output=True,
)

data = json.loads(kube_output.stdout)

output = []
for entry in data["items"]:
    if (kind := entry["kind"]) != "Job":
        print("Found kind", kind)
        continue
    status = entry["status"]
    metadata = entry["metadata"]
    labels = metadata["labels"]
    row = {
        "job-name": metadata.get("name"),
        "is_active": status.get("active"),
        "is_terminating": status.get("terminating"),
        "gpu-count": int(labels.get("fun.com/gpu-count", 0)),
        "user": labels.get("fun.com/user"),
    }
    output.append(row)

table = pd.DataFrame(output)

counts = table.groupby("user")["gpu-count"].sum().sort_values(ascending=False)
counts["total"] = counts.sum()

print("All Research Jobs")
print(table)
print()
print("By User")
print(counts.to_frame())
