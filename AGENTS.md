# Canonical checkout

`origin/main` on https://github.com/aleksandrsafiullin/Predictive-Maintenance-Lab is the only version of this project. Every other local commit, branch, and uncommitted edit to a tracked file is obsolete.

If this checkout is already on `origin/main` and `git status` shows no tracked changes, do nothing. Otherwise replace the checkout before any other work.

Do not merge, rebase, cherry-pick, or keep local commits, branches, or uncommitted edits to tracked files. Do not force-push.

From the repository root:

```bash
git fetch origin
git checkout main
git reset --hard origin/main
git branch --set-upstream-to=origin/main main
```

Confirm with:

```bash
git rev-parse HEAD
git status -sb
```

`HEAD` must equal `origin/main`. Status must show `main...origin/main` with no tracked changes.

Do not run `git clean -xfd`. Leave gitignored `data/`, `data/projects/`, `runs/`, and `.venv/` in place. After the reset, restart the app from this tree (`.venv/bin/python -m pdm app` or `./run.command`).
