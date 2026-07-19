from __future__ import annotations

import argparse
from pathlib import Path

from petra.segmentation.corpus import lint_corpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/validation/manifest.json")
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    errors = lint_corpus(Path(args.manifest), require_frozen=args.require_frozen)
    if errors:
        print("\n".join(errors))
        return 1
    print("Corpus de validacao consistente")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
