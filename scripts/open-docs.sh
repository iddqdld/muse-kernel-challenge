#!/bin/sh
# Serve the generated documentation on localhost and open it in the Windows
# browser, over the WSL2 localhost forward.
#
# Windows cannot open these pages straight from the Linux filesystem, and
# cmd.exe refuses UNC paths, so the pages are served from the repository root
# and opened by URL. Relative links between pages keep working.
#
# Usage: scripts/open-docs.sh [port]
# Without a port, the script uses the first free port from 8765.
# The server runs until you press Ctrl+C.

set -eu

# Exit 0 when a socket already listens on the port.
listening() {
  python3 -c 'import socket, sys; raise SystemExit(0 if socket.socket().connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$1" 2>/dev/null
}

if [ "$#" -ge 1 ]; then
  port=$1
  if listening "$port"; then
    echo "port ${port} is already in use" >&2
    exit 1
  fi
else
  port=
  for candidate in $(seq 8765 8795); do
    if ! listening "$candidate"; then
      port=$candidate
      break
    fi
  done
  if [ -z "$port" ]; then
    echo "no free port between 8765 and 8795" >&2
    exit 1
  fi
fi

repo=$(git rev-parse --show-toplevel)
page="docs/"
url="http://localhost:${port}/${page}"

cd "$repo"
python3 -m http.server "$port" --bind 127.0.0.1 >/dev/null 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT INT TERM

attempt=0
while [ "$attempt" -lt 50 ]; do
  if python3 -c 'import socket, sys; raise SystemExit(0 if socket.socket().connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$port" 2>/dev/null; then
    break
  fi
  attempt=$((attempt + 1))
  sleep 0.1
done

if [ "$attempt" -ge 50 ]; then
  echo "the page server did not start on port ${port}" >&2
  exit 1
fi

# cmd.exe needs a Windows working directory to accept a URL argument.
if cd /mnt/c 2>/dev/null; then
  cmd.exe /c start "" "$url" >/dev/null 2>&1 || true
fi

echo "serving ${repo} at http://localhost:${port}/"
echo "open ${url} in the Windows browser if it did not open automatically"
echo "press Ctrl+C to stop"
wait "$server"
