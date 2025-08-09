#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
DEST="wim:~/edm"       # scp target
DIST_DIR="dist"        # local output dir for tarballs
MARKER_SUFFIX="__COMMIT.txt"   # marker filename suffix

# ---- usage ----
if [ $# -lt 1 ]; then
  echo "Usage: $0 \"ship message here\""
  exit 2
fi
MSG="$1"

# ---- ensure git repo ----
git rev-parse --git-dir >/dev/null

# ---- commit ONLY tracked changes; abort if nothing to commit ----
# -a: commit all tracked modifications/deletions
# (won't include new untracked files unless you explicitly `git add`-ed them)
if ! git commit -am "$MSG" ; then
  echo "No tracked changes to commit (or commit failed). Aborting."
  exit 1
fi

# ---- push changes; abort if failed ----
if ! git push ; then
  echo "Push failed. Aborting."
  exit 1
fi

# ---- capture metadata ----
HASH=$(git rev-parse --short=8 HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROJECT=$(basename "$(git rev-parse --show-toplevel)")

# ---- build a clean export of EXACT commit (tracked files only) ----
# We use `git archive` so untracked files never sneak in.
TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

# Extract commit snapshot under a top-level project dir
git archive --format=tar --prefix="${PROJECT}/" HEAD | tar -x -C "$TMPDIR"

# ---- write marker file into the exported tree (not into your working tree) ----
MARKER_PATH="${TMPDIR}/${PROJECT}/${HASH}${MARKER_SUFFIX}"
{
  echo "project:  ${PROJECT}"
  echo "commit:   ${HASH}"
  echo "branch:   ${BRANCH}"
  echo "date_utc: ${STAMP}"
  echo "message:  ${MSG}"
} > "$MARKER_PATH"

# ---- pack tar.gz ----
mkdir -p "$DIST_DIR"
TARBALL="${DIST_DIR}/${PROJECT}-${HASH}-${STAMP}.tar.gz"
tar -C "$TMPDIR" -czf "$TARBALL" "${PROJECT}"

echo "Created: $TARBALL"

# ---- ship to cluster ----
scp "$TARBALL" "$DEST"
echo "Shipped to: $DEST"
