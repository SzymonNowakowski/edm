def get_git_commit_hash_from_marker(marker_path="__COMMIT.txt"):
    with open(marker_path, "r") as f:
        for line in f:
            if line.startswith("commit:"):
                return line.split(":", 1)[1].strip()
    raise ValueError("No 'commit:' line found in marker file")
