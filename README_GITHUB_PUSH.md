# Push Gov-Mem to GitHub

This document describes how to push the server copy of Gov-Mem to the GitHub
repository:

`git@github.com:jindongli-Ai/Gov-Mem.git`

The canonical server project directory is:

`/mnt/data_disk_2/fuyali/codes/2027_TOIS_Gov-Mem`

## 1. Enter the canonical project

```bash
cd /mnt/data_disk_2/fuyali/codes/2027_TOIS_Gov-Mem
```

Do not run the commands from the old migrated project directory or from a
different checkout.

## 2. Check the local repository

```bash
git status --short
git branch --show-current
git remote -v
git log -1 --oneline
```

The active branch should be `main`, and `origin` should point to
`git@github.com:jindongli-Ai/Gov-Mem.git`.

Review the changes before committing:

```bash
git diff
git diff --stat
```

Never commit API keys, private keys, `.env` files, caches, generated outputs,
or benchmark data that is excluded by `.gitignore`.

## 3. Commit the intended changes

```bash
git add -A
git status --short
git commit -m "Describe the change"
```

If there is nothing to commit, continue with the existing local commit. Do not
use `git reset --hard` to discard work.

## 4. Push with the Gov-Mem GitHub key

This checkout has a repository-level SSH configuration in `.git/config`:

```text
core.sshcommand=ssh -i /mnt/data_disk/home/fuyali/.ssh/id_ed25519_govmem_v2 -o IdentitiesOnly=yes
```

Use the repository configuration directly:

```bash
git push origin main
```

If the Git environment does not load the repository-level setting, specify the
same key explicitly:

```bash
GIT_SSH_COMMAND='ssh -i /mnt/data_disk/home/fuyali/.ssh/id_ed25519_govmem_v2 -o IdentitiesOnly=yes -o UserKnownHostsFile=/mnt/data_disk/home/fuyali/.ssh/known_hosts' \
  git push origin main
```

The private key must remain outside the repository. Do not print it, copy it
into the project, or add it to Git.

## 5. Verify the remote update

Check the commit pushed to GitHub:

```bash
git log -1 --format='%H %s'
git ls-remote origin main
git status --short
```

The hash printed by `git ls-remote origin main` must match the local commit
hash. The final `git status --short` should be empty unless there are
intentional uncommitted changes.

## 6. Test SSH authentication without pushing

```bash
GIT_SSH_COMMAND='ssh -i /mnt/data_disk/home/fuyali/.ssh/id_ed25519_govmem_v2 -o IdentitiesOnly=yes -o UserKnownHostsFile=/mnt/data_disk/home/fuyali/.ssh/known_hosts' \
  git ls-remote origin main
```

A successful command prints the remote `main` commit hash. The following
messages indicate authentication problems rather than code problems:

- `Host key verification failed`: the SSH known-hosts file is missing or does
  not contain GitHub's host key.
- `Permission denied (publickey)`: the configured key was not used, is not
  readable by the current user, or is not authorized for the GitHub account.
- `No anonymous write access`: HTTPS was used without a valid GitHub token.

Do not replace the remote with an unauthenticated HTTPS URL. Fix the SSH key or
use an approved GitHub token through the credential manager instead.

## Current repository state

The latest architecture organization commit is:

`248f273 docs: organize canonical Gov-Mem V8 architecture`

