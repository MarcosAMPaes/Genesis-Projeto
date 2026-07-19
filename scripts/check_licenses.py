from __future__ import annotations

from importlib import metadata

BLOCKED_MARKERS = ("AGPL", "SSPL", "BUSL", "NON-COMMERCIAL", "CC-BY-NC")


def main() -> int:
    blocked: list[str] = []
    inspected = 0
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "unknown")
        license_text = " ".join(
            filter(
                None,
                (
                    dist.metadata.get("License-Expression"),
                    dist.metadata.get("License"),
                    " ".join(dist.metadata.get_all("Classifier") or []),
                ),
            )
        ).upper()
        inspected += 1
        if any(marker in license_text for marker in BLOCKED_MARKERS):
            blocked.append(f"{name}: {license_text[:160]}")
    if blocked:
        print("Dependencias com licencas bloqueadas:")
        print("\n".join(blocked))
        return 1
    print(f"Licencas inspecionadas: {inspected}; bloqueios: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
