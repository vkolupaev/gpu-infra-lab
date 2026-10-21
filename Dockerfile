# syntax=docker/dockerfile:1.7
# Lab image = official vLLM image (pinned by digest) + thin tooling layer.
# Targets:  lab     -> runs + nsys CLI (small, default)
#           lab-ncu -> lab + Nsight Compute (heavy, only for VM providers with counters)
# Build:    docker build --target lab     -t ghcr.io/<user>/gpu-infra-lab:v0.32.0-cu129-<sha> .
#           docker build --target lab-ncu -t ghcr.io/<user>/gpu-infra-lab:v0.32.0-cu129-<sha>-ncu .

ARG VLLM_IMAGE=vllm/vllm-openai:v0.32.0-cu129@sha256:bc0445dac61accbf1c3b17fbb524a71eee0b1b32f55ba45952095e3be8991c3f

FROM ${VLLM_IMAGE} AS lab
ARG VLLM_IMAGE
# Exact versions from `apt-cache madison <pkg>` inside the base image (day 1).
ARG NSYS_PKG=cuda-nsight-systems-12-9
ARG NSYS_VERSION=12.9.2-1
ENV LAB_IMAGE=${VLLM_IMAGE} \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git openssh-server tmux jq ${NSYS_PKG}=${NSYS_VERSION} \
 && rm -rf /var/lib/apt/lists/*
COPY scripts/start.sh /usr/local/bin/lab-start
# The official image starts `vllm serve` with no args; a lab container must idle instead.
ENTRYPOINT []
CMD ["/usr/local/bin/lab-start"]

FROM lab AS lab-ncu
ARG NCU_PKG=cuda-nsight-compute-12-9
ARG NCU_VERSION=12.9.2-1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ${NCU_PKG}=${NCU_VERSION} \
 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:${PATH}
