#!/usr/bin/env bash
# One-time (or on rotation): save the GitHub token to a chmod-600 file on the persistent volume and test it. The token is never echoed.
set -u
export CG_ROOT=${CG_ROOT:-/workspace/vjepa-causal-geometry}
F="$CG_ROOT/secrets/gh_token"; mkdir -p "$(dirname "$F")"; chmod 700 "$(dirname "$F")"
read -rs -p "Paste token, then press Enter (nothing will show): " T; echo
T=$(printf %s "$T" | tr -d '\r\n \t\000-\037')
if [ -z "$T" ]; then echo "EMPTY - the paste did not arrive. Use Ctrl+Shift+V (or right-click > Paste) and try again."; exit 1; fi
printf %s "$T" > "$F"; chmod 600 "$F"; unset T
echo "saved: $F (len=$(wc -c < "$F") starts=$(head -c 11 "$F"))"
cd "$(dirname "$0")/.." && export GH_TOKEN=$(cat "$F")
if git -c credential.helper='!f(){ echo username=x-access-token; echo "password=$GH_TOKEN"; }; f' push --dry-run origin HEAD:main >/dev/null 2>/tmp/_tok_err; then echo "GitHub: token ACCEPTED for push"; else echo "GitHub: REJECTED -> $(tail -1 /tmp/_tok_err)"; echo "Fix: token expired/revoked, not scoped to V-JEPA-causal-geometry, or Contents is not 'Read and write'."; exit 2; fi
