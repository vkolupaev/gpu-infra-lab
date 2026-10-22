#!/usr/bin/env bash
# Container start: sshd (key auth only), then idle.
# Providers that inject PUBLIC_KEY (RunPod) get it appended to authorized_keys;
# others (Vast) write authorized_keys themselves.
ls /etc/ssh/ssh_host_*_key >/dev/null 2>&1 || ssh-keygen -A
mkdir -p ~/.ssh && chmod 700 ~/.ssh
AK=~/.ssh/authorized_keys
touch "$AK" && chmod 600 "$AK"
if [ -n "${PUBLIC_KEY:-}" ] && ! grep -qxF -- "$PUBLIC_KEY" "$AK"; then
  [ -s "$AK" ] && [ -n "$(tail -c1 "$AK")" ] && echo >> "$AK"   # keep keys on separate lines
  echo "$PUBLIC_KEY" >> "$AK"
fi
if ! grep -qE '^[^#[:space:]]' "$AK"; then
  echo "lab-start: WARNING: no keys in $AK, SSH login will fail unless the provider adds one" >&2
fi
mkdir -p /run/sshd && /usr/sbin/sshd \
  -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PermitRootLogin=prohibit-password
exec sleep infinity
