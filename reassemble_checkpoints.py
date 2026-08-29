from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "checkpoint_parts_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in data["files"]:
        target = ROOT / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            for relative_part in item["parts"]:
                with (ROOT / relative_part).open("rb") as source:
                    shutil.copyfileobj(source, output)
        actual_size = target.stat().st_size
        actual_hash = sha256(target)
        if actual_size != item["size"] or actual_hash != item["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Verification failed: {item['path']}")
        print(f"OK  {item['path']}  {actual_size} bytes")
    print(f"Restored and verified {len(data['files'])} checkpoints.")


if __name__ == "__main__":
    main()
