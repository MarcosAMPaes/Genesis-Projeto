from __future__ import annotations

import json
from pathlib import Path

from petra.contracts.schemas import SCHEMA_FACTORIES


def render_schema(schema: dict[str, object]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    schema_dir = Path("schemas")
    schema_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in sorted(SCHEMA_FACTORIES.items()):
        (schema_dir / name).write_text(render_schema(factory()), encoding="utf-8")
    print(f"Schemas gerados: {len(SCHEMA_FACTORIES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
