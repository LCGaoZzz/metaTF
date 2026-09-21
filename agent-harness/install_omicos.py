"""Install the portable metaTF Agent/Skill into an explicitly chosen local tree."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def install(destination: Path | str, layout: str = "workspace", dry_run: bool = False) -> dict:
    if layout not in {"workspace", "catalog"}:
        raise ValueError("layout must be workspace or catalog")
    source = Path(__file__).resolve().parent / "omicos"
    destination = Path(destination).resolve()
    root = destination / "domains" / "biology" if layout == "catalog" else destination
    pairs = [(source / "agents" / "metatf_analyst.md", root / "agents" / "metatf_analyst.md"),
             (source / "skills" / "metatf", root / "skills" / "metatf")]
    for src, target in pairs:
        if not src.exists():
            raise FileNotFoundError(f"Missing bundle resource: {src}")
        if src.is_dir() and target.resolve().is_relative_to(src.resolve()):
            raise ValueError("Destination cannot place the Skill inside its own source tree")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to replace existing content: {target}")
        parent = target.parent
        while parent != destination:
            if parent.is_symlink():
                raise ValueError(f"Refusing symlinked destination parent: {parent}")
            parent = parent.parent
    created = []
    if not dry_run:
        try:
            for src, target in pairs:
                target.parent.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    target.mkdir()  # exclusive reservation; never merge an existing Skill
                    created.append(target)
                    shutil.copytree(src, target, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                else:
                    with target.open("xb") as handle:
                        created.append(target)
                        handle.write(src.read_bytes())
        except Exception:
            for target in reversed(created):
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            raise
    return {"status": "planned" if dry_run else "installed", "layout": layout,
            "paths": [str(dst) for _, dst in pairs], "production_deployed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--layout", choices=["workspace", "catalog"], default="workspace")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(install(args.destination, args.layout, args.dry_run)))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
