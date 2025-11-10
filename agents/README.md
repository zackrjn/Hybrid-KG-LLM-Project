# Agents: Multi-Agent Workflow (Cursor 2.0)

This folder documents how to use Cursor 2.0 multi-agent mode safely with isolated git worktrees so multiple agents (Claude, GPT-5, Composer) can work concurrently without file conflicts.

## Roles

- Claude 4 Sonnet: analysis, reviews, KG reasoning
- GPT-5: long-context planning, cross-repo summaries, safety checks
- Composer (Cursor 2.0): execution agent for multi-file PyTorch/PyG generations

## Worktree Isolation

Create one worktree per agent/task to prevent conflicts. Use the provided scripts:

### Windows (PowerShell)

```powershell
.# Create a new worktree from `develop` for SNS task using Claude:
./scripts/dev/new_agent_worktree.ps1 -Task sns-ranking -Model claude-sonnet -BaseBranch develop

.# Open the new worktree folder in a new Cursor window and attach the agent to that window
```

### macOS/Linux (bash)

```bash
# Create a new worktree from `develop` for GraphViz augments using GPT-5
bash scripts/dev/new_agent_worktree.sh -t graphviz-aug -m gpt5 -b develop

# Open the new worktree folder in a new Cursor window and attach the agent
```

The script creates a branch named `agent/<task>-<model>-<timestamp>` and a corresponding folder under `worktrees/`.

## Branching Rules

- Name: `agent/<task>-<model>-<YYYYmmdd-HHMM>`
- PR flow: `agent/*` → `develop` → `main` after CI
- Ownership examples:
  - SNS agent: `scripts/prepare_hybrid_dataset.py`, `src/sns_ranker.py`
  - GraphWiz agent: `src/kg_visualize.py`, `graphwiz_module/*`

## Provider Setup

Copy `.env.example` to `.env` and fill in keys:

```
OPENAI_API_KEY=...
GROQ_API_KEY=...
OLLAMA_BASE_URL=http://127.0.0.1:11434
NGROK_AUTHTOKEN=...
COMPOSER_ENABLED=true
```

Use `agents/providers.py` to centralize client initialization. Models are routed by task at the call site.

## Cursor Multi-Agent Tips

- Prefer 2–3 agents by default; scale to 8 for batch sweeps.
- Use @folders in each worktree to limit context to that task’s directories.
- Use Plan Mode for high-impact changes; Composer executes only after plan approval.

## Reproducibility

- Commit run configs and manifests under `experiments/<task>/runs/<timestamp>/`.
- Keep deterministic seeds and cache dataset subsets under `data/primekg/cache/` (see plan).


