from __future__ import annotations

from jupyter_core.paths import jupyter_config_dir


def main() -> int:
    print(jupyter_config_dir())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
