from __future__ import annotations

import json
from pathlib import Path

from jsonschema.validators import validator_for


def main() -> int:
    schema_dir = Path("schemas")
    paths = sorted(schema_dir.glob("*.json")) if schema_dir.exists() else []
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        validator_for(schema).check_schema(schema)
    print(f"Schemas validos: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
