# Put mission_mujoco on GitHub

Run these commands in your regular WSL terminal, not inside the pyUR container.
The local project is `/home/harsh/mujoco_source_only/mission_mujoco`.
This guide publishes that project as its own repository; it does not include pyUR
or the other sibling projects in `mujoco_source_only`.

The project overview is in [README.md](README.md); running the simulation is
covered in [CLI_README.md](CLI_README.md).

## 1. Enter the project and initialize Git

```bash
cd /home/harsh/mujoco_source_only/mission_mujoco
git init
```

Do this initialization only once. For an existing repository, use `git status`
and continue from its current state instead.

## 2. Set your commit identity if needed

Check your existing identity:

```bash
git config user.name
git config user.email
```

If they are missing or incorrect, replace these example values and run:

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

These settings apply only to this repository. Your email becomes part of commit
metadata; you can use the GitHub-provided no-reply email from your account settings.
Git identity is separate from GitHub authentication.

## 3. Stage, inspect, and commit the project

```bash
git add .
git status --short
git diff --cached --stat
```

Check the listed files before committing. The existing `.gitignore` excludes
Python caches, `.venv`, build outputs, package metadata, and `results/local/`.
It does not ignore every results directory or notebook output. Do not commit
credentials or private data.

Then create the first commit and name the branch `main`:

```bash
git commit -m "Initial commit: mission-aligned MuJoCo planning and MPC"
git branch -M main
```

These commands also work with older Git versions that do not support
`git init -b main`.

## 4. Choose ONE remote setup method

### A. Create the GitHub repository from the command line

This route uses GitHub CLI (`gh`). Check whether it is installed:

```bash
gh --version
```

On Ubuntu/WSL, if needed:

```bash
sudo apt update
sudo apt install gh
```

Authenticate, configure Git authentication, then create and push a private repo:

```bash
gh auth login
gh auth setup-git
gh repo create mission_mujoco --private --source=. --remote=origin --push
```

Choose GitHub.com and follow the login prompts. The final command:

- Creates a remote repository named `mission_mujoco` in your GitHub account.
- Adds that repository as the local remote named `origin`.
- Pushes your committed branch.

Use `--public` instead of `--private` only if you want the repository public.
Do not use method B afterward: this command has already created the remote and
pushed the code. If the GitHub repository already exists, use method B instead.

### B. Connect an existing EMPTY remote repository URL

Use this route if you already have the remote URL. Replace the placeholder with
its actual URL:

```bash
REMOTE_URL='https://github.com/YOUR_USERNAME/mission_mujoco.git'
git remote add origin "$REMOTE_URL"
git remote -v
git push -u origin main
```

The remote repository should be empty: do not initialize it with an additional
README, license, or `.gitignore`, since those files already exist locally.

For GitHub HTTPS authentication, you can first use:

```bash
gh auth login
gh auth setup-git
```

Alternatively use your existing SSH or credential-manager setup. A GitHub account
password is not used as the HTTPS Git password.

If `origin` is already configured, inspect it first:

```bash
git remote -v
```

If you intentionally want to replace its URL:

```bash
git remote set-url origin "$REMOTE_URL"
```

If the remote already contains commits, inspect and reconcile its history before
pushing. Do not use a force push as a routine fix.

## 5. Verify the result

```bash
git remote -v
git status
git log -1 --oneline
```

If you used GitHub CLI, you can also inspect the remote:

```bash
gh repo view
```

`git status` should report a clean working tree after all intended changes are
committed. After `git push -u origin main`, the local main branch tracks the remote
main branch.

## 6. Save and upload later changes

From the project directory:

```bash
git status
git diff
git add .
git diff --cached --stat
git commit -m "Describe your change"
git push
```

For more precise commits, stage named files instead of everything, for example:

```bash
git add mission_mujoco/controller.py README.md
git commit -m "Update MPC controller and documentation"
git push
```

`git commit` saves a local version. `git push` uploads committed versions to the
remote. Editing a file does not automatically update GitHub.

## Command meanings

| Command | Meaning |
|---|---|
| `git init` | Create local Git metadata in this project. |
| `git add .` | Stage new/modified/deleted files under this directory, respecting ignore rules for untracked files. |
| `git commit -m "..."` | Save staged changes as a local commit. |
| `git branch -M main` | Rename the current branch to `main`. |
| `git remote add origin URL` | Associate a remote URL with the name `origin`. Does not upload files. |
| `git push -u origin main` | Upload `main` and establish its upstream tracking branch. |
| `git status` | Show branch and staged/unstaged file status. |
| `git diff` | Show unstaged edits to tracked files. |
| `git diff --cached` | Show staged changes that the next commit will include. |
| `gh repo create ...` | Create a repository on GitHub; the flags above also link and push the local repo. |

Official reference: [Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).
