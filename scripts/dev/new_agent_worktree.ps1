param(
    [string]$Task = "generic",
    [string]$Model = "claude-sonnet",
    [string]$BaseBranch = "develop"
)

$ErrorActionPreference = "Stop"

function Ensure-InGitRepo {
    if (-not (Test-Path ".git")) {
        throw "This script must be run at the repository root (where .git exists)."
    }
}

function Ensure-WorktreesDir {
    $wt = Join-Path -Path (Get-Location) -ChildPath "worktrees"
    if (-not (Test-Path $wt)) {
        New-Item -ItemType Directory -Path $wt | Out-Null
    }
    return $wt
}

function Get-Timestamp {
    return (Get-Date -Format "yyyyMMdd-HHmm")
}

Ensure-InGitRepo

# Verify base branch exists locally or fetch it
git rev-parse --verify $BaseBranch 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Base branch '$BaseBranch' not found locally; fetching..."
    git fetch origin $BaseBranch
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch base branch '$BaseBranch' from origin."
    }
}

$timestamp = Get-Timestamp
$branchName = "agent/$Task-$Model-$timestamp"
$worktreesDir = Ensure-WorktreesDir
$worktreePath = Join-Path $worktreesDir $branchName.Replace('/', '_')

# Create branch from base and worktree folder
git branch --no-track $branchName $BaseBranch
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create branch '$branchName' from '$BaseBranch'."
}

git worktree add "$worktreePath" $branchName
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create worktree at '$worktreePath' for branch '$branchName'."
}

Write-Host "Created worktree:" -ForegroundColor Green
Write-Host "  Branch:    $branchName"
Write-Host "  Path:      $worktreePath"
Write-Host "  Base:      $BaseBranch"
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1) Open '$worktreePath' in a new Cursor window"
Write-Host "  2) Attach your agent (Claude/GPT-5/Composer) to that window"
Write-Host "  3) Use @folders to limit context to task-specific dirs"


