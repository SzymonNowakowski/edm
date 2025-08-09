from pathlib import Path

def get_git_commit_hash_from_marker(marker_path: str | Path = "__COMMIT.txt") -> str:
    for line in Path(marker_path).read_text().splitlines():
        if line.startswith("commit:"):
            return line.split("commit:", 1)[1].strip()
    raise ValueError("No 'commit:' line found in marker file")
