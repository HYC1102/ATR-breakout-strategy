from pathlib import Path
import subprocess


MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024


def test_no_large_files_are_tracked():
    """Keep regenerable market-data caches and other large artifacts out of Git."""
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    ).decode().split("\0")
    oversized = []
    for relative in filter(None, output):
        path = root / relative
        if path.is_file() and path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            oversized.append(f"{relative} ({path.stat().st_size:,} bytes)")
    assert not oversized, "Large tracked files:\n" + "\n".join(oversized)
