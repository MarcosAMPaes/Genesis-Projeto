from __future__ import annotations

import json
from pathlib import Path

from jsonschema.validators import validator_for

from petra.contracts.schemas import SCHEMA_FACTORIES
from scripts.generate_schemas import render_schema


def main() -> int:
    schema_dir = Path("schemas")
    expected_names = set(SCHEMA_FACTORIES)
    paths = sorted(schema_dir.glob("*.json")) if schema_dir.exists() else []
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        print(f"Schemas ausentes: {missing}; inesperados: {unexpected}")
        return 1
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        validator_for(schema).check_schema(schema)
        expected = render_schema(SCHEMA_FACTORIES[path.name]())
        if path.read_text(encoding="utf-8") != expected:
            print(f"Schema desatualizado: {path}")
            return 1
    print(f"Schemas validos: {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
