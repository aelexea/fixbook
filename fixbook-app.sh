#!/bin/bash
# Starts the local Fixbook page (unless it is already running) and opens it in its own window.
# The port must match PORT in web.py.
URL=http://127.0.0.1:8765

cd "$(dirname "$(readlink -f "$0")")" || exit 1

if ! curl -s -o /dev/null --max-time 1 "$URL/api/state"; then
  setsid python3 web.py >>web.log 2>&1 &
  for _ in {1..30}; do
    curl -s -o /dev/null --max-time 1 "$URL/api/state" && break
    sleep 0.2
  done
fi

# omarchy-launch-webapp exists on Omarchy; anywhere else the default browser is used.
omarchy-launch-webapp "$URL" 2>/dev/null || xdg-open "$URL"
