import collections
import json
import subprocess
import typing
import re

import pandas as pd


def sanitize_usage_str(s: str) -> int:
    number = re.sub("[a-zA-Z]", "", s)
    return int(number)


def update_usage(total: dict[str, int], usage_dict: dict[str, str]) -> dict[str, int]:
    for k, v in usage_dict.items():
        total[k] += sanitize_usage_str(v)


def prepend_to_keys(dct: dict[str, typing.Any], prefix: str) -> dict[str, typing.Any]:
    return {f"{prefix}{k}": v for k, v in dct.items()}


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

    # Sum usage over containers
    containers = entry["spec"]["template"]["spec"]["containers"]
    requested = collections.defaultdict(int)
    limits = collections.defaultdict(int)
    for c in containers:
        resources = c["resources"]
        update_usage(requested, resources["requests"])
        update_usage(limits, resources["limits"])
    requested = prepend_to_keys(requested, "requested_")
    limits = prepend_to_keys(limits, "limits_")

    row = {
        "job-name": metadata.get("name"),
        "is_active": status.get("active"),
        "is_terminating": status.get("terminating"),
        "fun-gpu-count": int(labels.get("fun.com/gpu-count", 0)),
        "user": labels.get("fun.com/user"),
    }
    # If requested and/or limits have empty entries at this stage,
    # e.g. because the job didn't require and GPUs, then these
    # will show up as NaNs in the DataFrame below.
    row.update(requested)
    row.update(limits)
    output.append(row)

table = pd.DataFrame(output)

# As per the comment above, the summation may hit NaNs, but
# they're treated as 0 so it's all good.
counts = (
    table.groupby("user")[
        ["fun-gpu-count", "requested_nvidia.com/gpu", "limits_nvidia.com/gpu"]
    ]
    .sum()
    .astype(int)
    .sort_values(by="fun-gpu-count", ascending=False)
)

counts.loc["total", :] = counts.sum(axis=0)

print("All Research Jobs")
print(table)
print()
print("By User")
print(counts)
