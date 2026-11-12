#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Audit Spot TPU v6e (Trillium) quota and ct6e-standard-4t availability across GCP zones.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONES=(
  "us-west1-c"
  "us-central1-b"
  "us-east1-d"
  "us-east5-b"
  "us-east5-a"
  "asia-northeast1-b"
  "europe-west4-a"
)

echo "=================================================================="
echo " Spot TPU v6e (Trillium) Capacity & Quota Audit"
echo " Project: ${PROJECT}"
echo "=================================================================="

echo ""
echo "[1/2] Checking compute.googleapis.com/preemptible_tpu_v6e regional quotas..."
gcloud alpha services quota list \
  --service=compute.googleapis.com \
  --consumer="projects/${PROJECT}" \
  --filter="metric=compute.googleapis.com/preemptible_tpu_v6e" \
  --format=json | python3 -c '
import sys, json
data = json.load(sys.stdin)
for item in data:
    for lim in item.get("consumerQuotaLimits", []):
        for b in lim.get("quotaBuckets", []):
            dims = b.get("dimensions", {})
            reg = dims.get("region", "global")
            eff = b.get("effectiveLimit", "0")
            print(f"  Region: {reg:20s} | Spot TPU v6e Limit: {eff} chips")
'

echo ""
echo "[2/2] Checking ct6e-standard-4t (4 chips / 128GB HBM / 720GB RAM per VM) availability..."
for z in "${ZONES[@]}"; do
  mt=$(gcloud compute machine-types list --project="${PROJECT}" --zones="${z}" --filter="name=ct6e-standard-4t" --format="value(name,guestCpus,memoryMb)" 2>/dev/null || true)
  if [[ -n "${mt}" ]]; then
    echo "  Zone: ${z:18s} | AVAILABLE (${mt})"
  else
    echo "  Zone: ${z:18s} | NOT AVAILABLE"
  fi
done
