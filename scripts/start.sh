#!/usr/bin/env bash
# Container start: sshd (key auth only; host keys are generated at image build), then idle.
# Providers that inject PUBLIC_KEY (RunPod) get it appended to authorized_keys.
mkdir -p ~/.ssh && chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
fi
if ! grep -qE '^[^#[:space:]]' ~/.ssh/authorized_keys; then
  echo "lab-start: WARNING: no keys in ~/.ssh/authorized_keys, SSH login will fail" >&2
fi
mkdir -p /run/sshd && /usr/sbin/sshd \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PermitRootLogin=prohibit-password
exec sleep infinity
