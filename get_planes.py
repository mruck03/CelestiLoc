import os
import json
import boto3
from datetime import datetime
import re

bucket_name = "umich-3d-celest"
json_dir = "./plane-info-dump"
output_dir = "./filtered_images"
prefix = ""
timestamp_format = "%Y-%m-%dT%H%M%S%fZ"
timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{9}Z)")

s3 = boto3.client("s3")
os.makedirs(output_dir, exist_ok=True)

json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
print(f"Found {len(json_files)} JSON files.\n")

valid = False
clusters = []

for file in json_files:
    file_path = os.path.join(json_dir, file)
    try:
        with open(file_path, "r") as f:
            content = json.load(f)

        if isinstance(content, dict) and isinstance(content.get("data"), list) and len(content["data"]) > 0:
            if not valid:
                start_time = os.path.splitext(file)[0]
                valid = True
        else:
            if valid:
                end_time = os.path.splitext(file)[0]
                clusters.append((start_time, end_time))
                valid = False

    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

# edge case: last file is valid and not closed out
if valid:
    clusters.append((start_time, os.path.splitext(json_files[-1])[0]))

print(f"Found {len(clusters)} clusters.\n")

paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

all_jpegs = []
for page in pages:
    for obj in page.get("Contents", []):
        key = obj["Key"]
        match = timestamp_pattern.search(key)
        if match:
            try:
                jpeg_time = datetime.strptime(match.group(1), timestamp_format)
                all_jpegs.append((jpeg_time, key))
            except Exception:
                continue

for i, (start_str, end_str) in enumerate(clusters):
    cluster_dir = os.path.join(output_dir, f"cluster_{i+1}")
    os.makedirs(cluster_dir, exist_ok=True)

    try:
        start_dt = datetime.strptime(start_str, timestamp_format)
        end_dt = datetime.strptime(end_str, timestamp_format)
    except Exception as e:
        print(f"Skipping cluster {i+1} due to timestamp parse error: {e}")
        continue

    print(f"Cluster {i+1}: {start_dt} → {end_dt}")

    for jpeg_time, key in all_jpegs:
        if start_dt <= jpeg_time <= end_dt:
            dest_path = os.path.join(cluster_dir, os.path.basename(key))
            print(f"Downloading: {key}")
            try:
                s3.download_file(bucket_name, key, dest_path)
            except Exception as e:
                print(f"Failed to download {key}: {e}")

print("\nDone!")