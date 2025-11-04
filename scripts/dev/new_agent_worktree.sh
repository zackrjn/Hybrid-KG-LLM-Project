#!/usr/bin/env bash
set -euo pipefail

TASK="generic"
MODEL="claude-sonnet"
BASE_BRANCH="develop"

usage() {
  echo "Usage: $0 [-t TASK] [-m MODEL] [-b BASE_BRANCH]" >&2
}

while getopts ":t:m:b:h" opt; do
  case $opt in
    t) TASK="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    b) BASE_BRANCH="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) usage; exit 1 ;;
  esac
done

if [ ! -d .git ]; then
  echo "This script must be run at the repository root (where .git exists)." >&2
  exit 1
fi

if ! git rev-parse --verify "$BASE_BRANCH" >/dev/null 2>&1; then
  echo "Base branch '$BASE_BRANCH' not found locally; fetching..."
  git fetch origin "$BASE_BRANCH"
fi

timestamp=$(date +%Y%m%d-%H%M)
branchName="agent/${TASK}-${MODEL}-${timestamp}"
worktreesDir="$(pwd)/worktrees"
mkdir -p "$worktreesDir"
worktreePath="${worktreesDir}/${branchName//\//_}"

git branch --no-track "$branchName" "$BASE_BRANCH"
git worktree add "$worktreePath" "$branchName"

echo "Created worktree:"
echo "  Branch:    $branchName"
echo "  Path:      $worktreePath"
echo "  Base:      $BASE_BRANCH"
echo "Next steps:"
echo "  1) Open '$worktreePath' in a new Cursor window"
echo "  2) Attach your agent (Claude/GPT-5/Composer) to that window"
echo "  3) Use @folders to limit context to task-specific dirs"


