"""Set up the Git repository with initial structure and commit."""
from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    """Run a subprocess command, raising an error if it fails."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_git_repo() -> None:
    git_dir = REPO_ROOT / ".git"
    if not git_dir.exists():
        run(["git", "init"])
    else:
        print("Git repository already initialized.")


def ensure_readme() -> None:
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists() or readme_path.read_text().strip() == "":
        readme_content = "# ISEF 2.0 Data: Alzheimer–Microbiome Raw Datasets\n"
    else:
        readme_content = readme_path.read_text()
        if "Alzheimer" not in readme_content:
            readme_content = "# ISEF 2.0 Data: Alzheimer–Microbiome Raw Datasets\n" + readme_content
    readme_path.write_text(readme_content)


def ensure_gitignore() -> None:
    gitignore_path = REPO_ROOT / ".gitignore"
    lines = [
        "__pycache__/",
        "*.pyc",
        ".DS_Store",
        "*.log",
        "*.tmp",
    ]
    gitignore_path.write_text("\n".join(lines) + "\n")


def initial_commit() -> None:
    run(["git", "add", "."])
    run([
        "git",
        "commit",
        "-m",
        "Initial commit: add Alzheimer–microbiome dataset scripts and structure",
    ])


def main() -> None:
    ensure_git_repo()
    ensure_readme()
    ensure_gitignore()
    initial_commit()


if __name__ == "__main__":
    main()
