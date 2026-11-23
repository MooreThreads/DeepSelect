#!/usr/bin/env bash
# Register this MUSA host as a repo-scoped GitHub Actions self-hosted runner for
# automatic deepselect-musa wheel builds (see .github/workflows/release-musa-auto.yml).
#
# The runner needs: docker (with MUSA runtime), MUSA driver, git, curl, systemd,
# and sudo for installing the systemd service.
#
# Usage:
#   sudo -v   # cache sudo credentials first
#   GITHUB_PAT=<token with repo admin> \
#     ./scripts/ci/musa/setup_github_runner.sh
#
# Optional environment:
#   GITHUB_REPO        default: MooreThreads/DeepSelect
#   RUNNER_DIR         default: /data/actions-runner-deepselect
#   RUNNER_NAME        default: ds-musa-$(hostname -s)
#   RUNNER_LABELS      default: musa,musa52,s5000,py310
#   RUNNER_WORK_DIR    default: $RUNNER_DIR/_work
#
# The token is only used to fetch a short-lived registration token and is never
# written to disk.
set -euo pipefail

REPO="${GITHUB_REPO:-MooreThreads/DeepSelect}"
RUNNER_DIR="${RUNNER_DIR:-/data/actions-runner-deepselect}"
RUNNER_NAME="${RUNNER_NAME:-ds-musa-$(hostname -s)}"
RUNNER_LABELS="${RUNNER_LABELS:-musa,musa52,s5000,py310}"
RUNNER_WORK_DIR="${RUNNER_WORK_DIR:-$RUNNER_DIR/_work}"

log() { printf '\n=== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(id -u)" -ne 0 ]] || die "run as the service user (e.g. mccxadmin), not root"
[[ -n "${GITHUB_PAT:-}" ]] || die "set GITHUB_PAT to a token with admin access to $REPO"
for c in curl git python3 docker systemctl; do command -v "$c" >/dev/null || die "missing required command: $c"; done
sudo -n true 2>/dev/null || die "passwordless sudo required for the systemd service; run 'sudo -v' first"

log "fetching runner registration token for $REPO"
TOKEN=$(curl -fsS -X POST \
  -H "Authorization: Bearer $GITHUB_PAT" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/$REPO/actions/runners/registration-token" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
[[ -n "$TOKEN" ]] || die "could not obtain a registration token"

log "downloading the latest actions/runner"
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"
ASSET_URL=$(python3 - <<'PY'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://api.github.com/repos/actions/runner/releases/latest"))
for a in d["assets"]:
    if a["name"].startswith("actions-runner-linux-x64-") and a["name"].endswith(".tar.gz"):
        print(a["browser_download_url"]); break
PY
)
[[ -n "$ASSET_URL" ]] || die "could not resolve the actions/runner asset URL"
curl -fsSL -o runner.tar.gz "$ASSET_URL"
tar xzf runner.tar.gz
rm -f runner.tar.gz

log "configuring runner $RUNNER_NAME (labels: $RUNNER_LABELS)"
./config.sh --unattended --replace \
  --url "https://github.com/$REPO" \
  --token "$TOKEN" \
  --name "$RUNNER_NAME" \
  --labels "$RUNNER_LABELS" \
  --work "$RUNNER_WORK_DIR"

log "installing systemd service (user: $(id -un))"
sudo ./svc.sh install "$(id -un)"
sudo ./svc.sh start
sudo ./svc.sh status || true

log "runner registered; check status with:"
printf '  sudo %s/svc.sh status\n  GH_TOKEN=<token> gh api repos/%s/actions/runners\n' "$RUNNER_DIR" "$REPO"
