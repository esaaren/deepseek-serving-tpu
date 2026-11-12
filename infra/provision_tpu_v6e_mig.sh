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

# Provision a Multi-Host TPU v6e (Trillium) Slice using GCE Native VMs,
# High-Throughput Workload Policy, and Managed Instance Group (MIG).
#
# Default configuration for DeepSeek-V4.1-Flash (552B total params):
#   - 8x ct6e-standard-4t VMs = 32 TPU v6e chips (1,024 GB HBM + 5,760 GB Host DRAM)
#   - 2D Torus Topology: 4x8 (TORCH_TPU_TOPOLOGY=4,8,1)
#   - Zone: us-west1-c (Spot)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-west1-c}"
REGION="${ZONE%-*}"
BASE_NAME="${BASE_NAME:-dsv41-v6e32}"
MACHINE_TYPE="${MACHINE_TYPE:-ct6e-standard-4t}"
TOPOLOGY="${TOPOLOGY:-4x8}"
TARGET_SIZE="${TARGET_SIZE:-8}"
SPOT="${SPOT:-true}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-$([[ "${SPOT}" == "true" ]] && echo "SPOT" || echo "FLEX_START")}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-4h}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-250GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
NETWORK="${NETWORK:-default}"
IMAGE_NAME="${IMAGE_NAME:-ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v20261016}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
INVENTORY_OUT="${INVENTORY_OUT:-${SCRIPT_DIR}/nodes_inventory.json}"

WP_NAME="${BASE_NAME}-wp"
TPL_NAME="${BASE_NAME}-tpl"
MIG_NAME="${BASE_NAME}-mig"

echo "=================================================================="
echo " Provisioning Multi-Host TPU v6e Slice via GCE MIG"
echo "------------------------------------------------------------------"
echo " Project      : ${PROJECT}"
echo " Zone         : ${ZONE} (Region: ${REGION})"
echo " Machine Type : ${MACHINE_TYPE} (4 TPU v6e chips / 128 GB HBM / 720 GB RAM per host)"
echo " Target Hosts : ${TARGET_SIZE} VMs (${TOPOLOGY} topology = $((TARGET_SIZE * 4)) TPU v6e chips)"
echo " Spot Mode    : ${SPOT}"
echo " Boot Disk    : ${BOOT_DISK_SIZE} ${BOOT_DISK_TYPE}"
echo " Image        : ${IMAGE_NAME}"
echo "=================================================================="

# 1. Create Workload Policy
if [[ -n "${TOPOLOGY}" && "${TOPOLOGY}" != "none" && "${MACHINE_TYPE}" != "ct6e-standard-8t" && "${TARGET_SIZE}" -gt 1 ]]; then
  echo "[1/4] Creating Workload Policy ${WP_NAME} (topology=${TOPOLOGY})..."
  if ! gcloud compute resource-policies describe "${WP_NAME}" --project="${PROJECT}" --region="${REGION}" >/dev/null 2>&1; then
    gcloud compute resource-policies create workload-policy "${WP_NAME}" \
      --project="${PROJECT}" \
      --region="${REGION}" \
      --type=HIGH_THROUGHPUT \
      --accelerator-topology="${TOPOLOGY}"
  else
    echo "      Workload policy ${WP_NAME} already exists."
  fi
fi

# 2. Create Instance Template
echo "[2/4] Creating Instance Template ${TPL_NAME}..."
if ! gcloud compute instance-templates describe "${TPL_NAME}" --project="${PROJECT}" >/dev/null 2>&1; then
  PROV_ARGS=(
    --project="${PROJECT}"
    --machine-type="${MACHINE_TYPE}"
    --network="${NETWORK}"
    --region="${REGION}"
    --image="${IMAGE_NAME}"
    --image-project="${IMAGE_PROJECT}"
    --boot-disk-size="${BOOT_DISK_SIZE}"
    --boot-disk-type="${BOOT_DISK_TYPE}"
    --maintenance-policy=TERMINATE
    --scopes=https://www.googleapis.com/auth/cloud-platform
  )
  if [[ "${PROVISIONING_MODEL}" == "FLEX_START" ]]; then
    PROV_ARGS+=(
      --provisioning-model=FLEX_START
      --max-run-duration="${MAX_RUN_DURATION}"
      --instance-termination-action=DELETE
      --reservation-affinity=none
    )
  elif [[ "${PROVISIONING_MODEL}" == "SPOT" ]]; then
    PROV_ARGS+=(
      --provisioning-model=SPOT
      --instance-termination-action=STOP
    )
  fi
  gcloud beta compute instance-templates create "${TPL_NAME}" "${PROV_ARGS[@]}"
else
  echo "      Instance template ${TPL_NAME} already exists."
fi

# 3. Create Managed Instance Group (MIG)
echo "[3/4] Creating Managed Instance Group ${MIG_NAME} (size=${TARGET_SIZE})..."
if ! gcloud compute instance-groups managed describe "${MIG_NAME}" --project="${PROJECT}" --region="${REGION}" >/dev/null 2>&1; then
  if [[ -n "${TOPOLOGY}" && "${TOPOLOGY}" != "none" && "${MACHINE_TYPE}" != "ct6e-standard-8t" && "${TARGET_SIZE}" -gt 1 ]]; then
    gcloud beta compute instance-groups managed create "${MIG_NAME}" \
      --project="${PROJECT}" \
      --base-instance-name="${BASE_NAME}" \
      --region="${REGION}" \
      --zones="${ZONE}" \
      --target-distribution-shape=any-single-zone \
      --instance-redistribution-type=none \
      --target-size-policy-mode=bulk \
      --default-action-on-vm-failure=do-nothing \
      --size="${TARGET_SIZE}" \
      --template="${TPL_NAME}" \
      --workload-policy="regions/${REGION}/resourcePolicies/${WP_NAME}"
  else
    gcloud beta compute instance-groups managed create "${MIG_NAME}" \
      --project="${PROJECT}" \
      --base-instance-name="${BASE_NAME}" \
      --region="${REGION}" \
      --zones="${ZONE}" \
      --target-distribution-shape=any-single-zone \
      --instance-redistribution-type=none \
      --default-action-on-vm-failure=do-nothing \
      --size="${TARGET_SIZE}" \
      --template="${TPL_NAME}"
  fi
else
  echo "      MIG ${MIG_NAME} already exists."
fi

# 4. Wait for all instances to reach RUNNING state and extract inventory
echo "[4/4] Waiting for all ${TARGET_SIZE} TPU v6e VMs to reach RUNNING state..."
for attempt in $(seq 1 60); do
  RUNNING_COUNT=$(gcloud compute instances list \
    --project="${PROJECT}" \
    --filter="name ~ ^${BASE_NAME}- AND zone:(${ZONE}) AND status=RUNNING" \
    --format="value(name)" | wc -l | tr -d ' ')
  echo "      Attempt ${attempt}/60: ${RUNNING_COUNT}/${TARGET_SIZE} VMs RUNNING..."
  if [[ "${RUNNING_COUNT}" -eq "${TARGET_SIZE}" ]]; then
    break
  fi
  if [[ "${attempt}" -eq 60 ]]; then
    echo "ERROR: Timed out waiting for ${TARGET_SIZE} VMs to reach RUNNING state." >&2
    gcloud compute instance-groups managed list-instances "${MIG_NAME}" --project="${PROJECT}" --zone="${ZONE}" >&2 || true
    exit 1
  fi
  sleep 10
done

gcloud compute instances list \
  --project="${PROJECT}" \
  --filter="name ~ ^${BASE_NAME}- AND zone:(${ZONE}) AND status=RUNNING" \
  --format="json(name,networkInterfaces[0].networkIP,networkInterfaces[0].accessConfigs[0].natIP,resourceStatus)" \
  | python3 -c '
import sys, json
data = json.load(sys.stdin)
data.sort(key=lambda x: x["name"])
nodes = []
for idx, item in enumerate(data):
    nodes.append({
        "rank": idx,
        "name": item["name"],
        "internal": item["networkInterfaces"][0]["networkIP"],
        "nat": item["networkInterfaces"][0]["accessConfigs"][0]["natIP"],
    })
print(json.dumps(nodes, indent=2))
' > "${INVENTORY_OUT}"

echo "Saved inventory (${TARGET_SIZE} hosts) to ${INVENTORY_OUT}:"
cat "${INVENTORY_OUT}"

# Clean stale SSH known_hosts entries for recycled Spot NAT IPs
python3 - "${INVENTORY_OUT}" <<'PY'
import json, subprocess, sys
with open(sys.argv[1]) as f:
    nodes = json.load(f)
for n in nodes:
    for ip in (n.get("nat"), n.get("internal")):
        if ip:
            subprocess.run(["ssh-keygen", "-R", ip], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
PY
echo "Cleaned stale SSH host keys for provisioned nodes."
