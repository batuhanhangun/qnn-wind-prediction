# Git provenance, read on the host. Sourced by every job script:
#
#   source "$QNNWIND_REPO/slurm/git_provenance.sh" strict   # tuning, QNN, classical jobs: refuse a dirty tree
#   source "$QNNWIND_REPO/slurm/git_provenance.sh" warn     # smoke, calibration: warn only
#
# Reads the commit and `git status --porcelain` of the clone at $QNNWIND_REPO, writes both to
# the job log, and exports QNNWIND_GIT_COMMIT, QNNWIND_GIT_DIRTY (0/1), and QNNWIND_GIT_STATUS
# (porcelain lines joined by "; "), which slurm/in_container.sh passes into the container and
# every result.json records. The container itself cannot run git reliably.

qnnwind_provenance_mode="${1:-warn}"
QNNWIND_GIT_COMMIT="$(git -C "$QNNWIND_REPO" rev-parse HEAD)"
qnnwind_git_status="$(git -C "$QNNWIND_REPO" status --porcelain)"
QNNWIND_GIT_STATUS="$(printf '%s' "$qnnwind_git_status" | awk 'NR > 1 { printf "; " } { printf "%s", $0 }')"
if [ -n "$qnnwind_git_status" ]; then QNNWIND_GIT_DIRTY=1; else QNNWIND_GIT_DIRTY=0; fi
export QNNWIND_GIT_COMMIT QNNWIND_GIT_DIRTY QNNWIND_GIT_STATUS

echo "git commit: $QNNWIND_GIT_COMMIT ($QNNWIND_REPO)"
if [ "$QNNWIND_GIT_DIRTY" = 1 ]; then
    echo "git working tree: dirty"
    printf '%s\n' "$qnnwind_git_status" | sed 's/^/    /'
    if [ "$qnnwind_provenance_mode" = strict ]; then
        echo "ERROR: the working tree of $QNNWIND_REPO is dirty; commit or discard the" \
             "changes before a tuning, QNN, or classical job." >&2
        return 1 2>/dev/null || exit 1
    fi
    echo "WARNING: dirty working tree; results record QNNWIND_GIT_DIRTY=1." >&2
else
    echo "git working tree: clean"
fi
