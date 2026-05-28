# One-time repo cleanup

Your `.gitignore` now excludes `__pycache__/` and `.DS_Store`, but those files
were committed **before** the ignore rules existed, so they're still tracked in
the repo. Run this once, locally, from the repo root to untrack them. Files
stay on disk — only the git index is updated.

```bash
# Untrack all __pycache__ directories already in the repo
git rm -r --cached --ignore-unmatch \
  sherloc/__pycache__ \
  sherloc/ctDNA_Dynamics_Encoder/__pycache__ \
  sherloc/Genomic_Interaction_Encoder/__pycache__ \
  scripts/__pycache__

# Untrack all .DS_Store files
git ls-files -z -- '**/.DS_Store' '.DS_Store' | xargs -0 git rm --cached --ignore-unmatch

# Commit the cleanup
git add .gitignore
git commit -m "chore: untrack __pycache__ and .DS_Store; broaden .gitignore"
```

If `find` is more convenient on your system:

```bash
find . -name __pycache__ -type d -exec git rm -r --cached --ignore-unmatch {} +
find . -name .DS_Store -exec git rm --cached --ignore-unmatch {} +
```

## Optional: drop them from the entire git history

The commands above stop tracking these files going forward but leave them in
historical commits, which still bloats `git clone`. If you care about cleaning
history (note: this rewrites commit SHAs — force-push and warn collaborators):

```bash
pip install git-filter-repo
git filter-repo --path-glob '**/__pycache__/**' --invert-paths
git filter-repo --path-glob '**/.DS_Store'      --invert-paths
git push --force origin main
```

## Sanity check after cleanup

```bash
git status                       # should show only the new/modified files
git ls-files | grep __pycache__  # should print nothing
git ls-files | grep .DS_Store    # should print nothing
```

Delete this file once you've run through it — it's just a one-time checklist.
