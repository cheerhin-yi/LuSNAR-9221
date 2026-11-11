#!/usr/bin/env bash
set -euo pipefail
HOST=$(grep -m1 nameserver /etc/resolv.conf | awk '{print $2}')
echo "HOST=$HOST"
ip route | head -5 || true
GW=$(ip route | awk '/default/{print $3; exit}')
echo "GW=$GW"
for h in "$HOST" "$GW"; do
  [ -z "$h" ] && continue
  echo "TRY $h:7897"
  if timeout 3 bash -c "echo >/dev/tcp/$h/7897" 2>/dev/null; then
    echo OPEN
  else
    echo CLOSED
  fi
done
echo '--- curl anaconda ---'
timeout 15 curl -I -sS https://repo.anaconda.com 2>&1 | head -8 || true
echo '--- curl pypi ---'
timeout 15 curl -I -sS https://pypi.org 2>&1 | head -8 || true
echo '--- curl via host proxy ---'
if [ -n "$HOST" ]; then
  timeout 15 curl -I -sS -x "http://$HOST:7897" https://repo.anaconda.com 2>&1 | head -8 || true
fi