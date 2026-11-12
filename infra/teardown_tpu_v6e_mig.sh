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

# Tear down Multi-Host TPU v6e MIG, Instance Template, and Workload Policy.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
ZONE="${ZONE:-us-west1-c}"
REGION="${ZONE%-*}"
BASE_NAME="${BASE_NAME:-dsv41-v6e32}"

WP_NAME="${BASE_NAME}-wp"
TPL_NAME="${BASE_NAME}-tpl"
MIG_NAME="${BASE_NAME}-mig"

echo "=================================================================="
echo " Tearing Down Multi-Host TPU v6e Slice (${BASE_NAME})"
echo " Project: ${PROJECT} | Zone: ${ZONE}"
echo "=================================================================="

echo "[1/3] Deleting Managed Instance Group ${MIG_NAME}..."
if gcloud compute instance-groups managed describe "${MIG_NAME}" --project="${PROJECT}" --region="${REGION}" >/dev/null 2>&1; then
  gcloud compute instance-groups managed delete "${MIG_NAME}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --quiet
elif gcloud compute instance-groups managed describe "${MIG_NAME}" --project="${PROJECT}" --zone="${ZONE}" >/dev/null 2>&1; then
  gcloud compute instance-groups managed delete "${MIG_NAME}" \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --quiet
else
  echo "      MIG ${MIG_NAME} not found."
fi

echo "[2/3] Deleting Instance Template ${TPL_NAME}..."
if gcloud compute instance-templates describe "${TPL_NAME}" --project="${PROJECT}" >/dev/null 2>&1; then
  gcloud compute instance-templates delete "${TPL_NAME}" \
    --project="${PROJECT}" \
    --quiet
else
  echo "      Instance template ${TPL_NAME} not found."
fi

echo "[3/4] Deleting Workload Policy ${WP_NAME}..."
if gcloud compute resource-policies describe "${WP_NAME}" --project="${PROJECT}" --region="${REGION}" >/dev/null 2>&1; then
  gcloud compute resource-policies delete "${WP_NAME}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --quiet
else
  echo "      Workload policy ${WP_NAME} not found."
fi

echo "[4/4] Deleting any standalone ${BASE_NAME}-* TPU instances in ${ZONE}..."
STANDALONE_VMS=$(gcloud compute instances list \
  --project="${PROJECT}" \
  --zones="${ZONE}" \
  --filter="name ~ ^${BASE_NAME}-" \
  --format="value(name)" | tr '\n' ' ')
if [[ -n "${STANDALONE_VMS// /}" ]]; then
  # shellcheck disable=SC2086
  gcloud compute instances delete ${STANDALONE_VMS} \
    --project="${PROJECT}" \
    --zone="${ZONE}" \
    --quiet
else
  echo "      No standalone ${BASE_NAME}-* instances found."
fi

echo "Verifying remaining TPU VMs in project ${PROJECT}:"
gcloud compute instances list --project="${PROJECT}" --filter="machineType ~ ct6e OR machineType ~ ct5"
echo "Teardown complete."
