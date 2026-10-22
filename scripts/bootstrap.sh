#!/usr/bin/env bash
# Fallback when the lab image is not in a registry yet: run once inside the OFFICIAL vLLM image.
# Keep package pins in sync with the Dockerfile.
set -euo pipefail
NSYS_PKG=${NSYS_PKG:-cuda-nsight-systems-12-9}
NSYS_VERSION=${NSYS_VERSION:?set the same pin as in Dockerfile}
apt-get update
apt-get install -y --no-install-recommends git tmux jq "${NSYS_PKG}=${NSYS_VERSION}"
# No lab image here: LAB_IMAGE says so explicitly (bench.py will warn), the base is recorded separately.
echo "export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE:-vllm/vllm-openai:v0.32.0-cu129}" \
  "LAB_IMAGE='${LAB_IMAGE:-none (bootstrap.sh)}'" >> ~/.bashrc
