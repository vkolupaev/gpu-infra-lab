# syntax=docker/dockerfile:1.7
# Lab image = official vLLM image (pinned by digest) + thin tooling layer.
# Targets:  lab     -> runs + nsys CLI (small, default)
#           lab-ncu -> lab + Nsight Compute (heavy, only for VM providers with counters)
# Build:    TAG=ghcr.io/vkolupaev/gpu-infra-lab:v0.32.0-cu129-$(git rev-parse --short HEAD)
#           docker build --target lab     --build-arg LAB_IMAGE=$TAG     -t $TAG .
#           docker build --target lab-ncu --build-arg LAB_IMAGE=$TAG-ncu -t $TAG-ncu .
# VLLM_BASE_IMAGE = what we build FROM; LAB_IMAGE = this image's own ref (bench.py records both).
# Our digest is unknown before push: on the host, export LAB_IMAGE=<tag>@<digest> (notes/images.md).

ARG VLLM_BASE_IMAGE=vllm/vllm-openai:v0.32.0-cu129@sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f

FROM ${VLLM_BASE_IMAGE} AS lab
ARG VLLM_BASE_IMAGE
# Exact versions from `apt-cache madison <pkg>` inside the base image (day 1).
ARG NSYS_PKG=cuda-nsight-systems-12-9
ARG NSYS_VERSION=12.9.2-1
ENV VLLM_BASE_IMAGE=${VLLM_BASE_IMAGE} \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git openssh-server tmux jq ${NSYS_PKG}=${NSYS_VERSION} \
 && rm -rf /var/lib/apt/lists/*
# 2026-10-25: the base image ships a cu13 torchcodec build in a cu129 image (libnvrtc.so.13
# missing), so `import vllm.multimodal` fails and `vllm serve` exits. Text-only models do not
# need torchcodec.
RUN python3 -m pip uninstall -y torchcodec
# Fail the build, not the rented host, when vLLM does not import.
RUN python3 -c "import torch, vllm, vllm.multimodal; print(vllm.__version__, torch.__version__, torch.version.cuda)"
COPY scripts/start.sh /usr/local/bin/lab-start
# The official image starts `vllm serve` with no args; a lab container must idle instead.
ENTRYPOINT []
CMD ["/usr/local/bin/lab-start"]
# Last, so a new git sha in the tag does not invalidate the apt layers.
ARG LAB_IMAGE=
ENV LAB_IMAGE=${LAB_IMAGE}

FROM lab AS lab-ncu
ARG NCU_PKG=cuda-nsight-compute-12-9
ARG NCU_VERSION=12.9.2-1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ${NCU_PKG}=${NCU_VERSION} \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:${PATH}
ARG LAB_IMAGE=
ENV LAB_IMAGE=${LAB_IMAGE}
