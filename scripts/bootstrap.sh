#!/usr/bin/env bash
# Fallback when the lab image is not in a registry yet: run once inside the OFFICIAL vLLM image.
# Keep package pins in sync with the Dockerfile.
set -euo pipefail
NSYS_PKG=${NSYS_PKG:-nsight-systems-cli}
NSYS_VERSION=${NSYS_VERSION:?set the same pin as in Dockerfile}
apt-get update
apt-get install -y --no-install-recommends git tmux jq "${NSYS_PKG}=${NSYS_VERSION}"
echo "export VLLM_WORKER_MULTIPROC_METHOD=spawn LAB_IMAGE=${LAB_IMAGE:-vllm/vllm-openai:v0.31.0-cu129+bootstrap}" >> ~/.bashrc
