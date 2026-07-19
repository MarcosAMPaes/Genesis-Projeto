from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from petra.contracts.schemas import SCHEMA_FACTORIES


@pytest.mark.contract
def test_generated_schemas_are_valid_and_current() -> None:
    for name, factory in SCHEMA_FACTORIES.items():
        schema = json.loads((Path("schemas") / name).read_text(encoding="utf-8"))
        validator_for(schema).check_schema(schema)
        assert schema == factory()
