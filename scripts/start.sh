#!/usr/bin/env bash
# Container start: optional sshd for providers that inject PUBLIC_KEY (RunPod), then idle.
if [ -n "${PUBLIC_KEY:-}" ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  mkdir -p /run/sshd && /usr/sbin/sshd
fi
exec sleep infinity
