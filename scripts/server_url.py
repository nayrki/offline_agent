#!/usr/bin/env python3
"""Get or set ``[remote].base_url`` in an offline-agent TOML config.

Shared by install.sh and install.bat so the read/edit logic lives in one place
(batch has no heredocs). Stdlib only; no offline_agent import, so it runs before
the package is installed and on any supported interpreter.

Usage:
    python scripts/server_url.py --get CONFIG          # print current value (or "")
    python scripts/server_url.py --set CONFIG URL      # register URL, preserving comments
"""
import re
import sys


def get(path: str) -> str:
    # tomllib is in the standard library on Python 3.11+, which matches the
    # package's minimum supported version.
    try:
        import tomllib
    except ModuleNotFoundError:
        return ""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f).get("remote", {}).get("base_url", "")
    except Exception:
        return ""


def set_url(path: str, url: str) -> None:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    def section_of(idx: int):
        sec = None
        for j in range(idx + 1):
            s = lines[j].strip()
            if s.startswith("[") and s.endswith("]"):
                sec = s
        return sec

    for i, line in enumerate(lines):
        if re.match(r"\s*base_url\s*=", line) and section_of(i) == "[remote]":
            m = re.search(r"(#.*)$", line.rstrip("\n"))
            comment = "  " + m.group(1) if m else ""
            lines[i] = f'base_url = "{url}"{comment}\n'
            break
    else:
        for i, line in enumerate(lines):
            if line.strip() == "[remote]":
                lines.insert(i + 1, f'base_url = "{url}"\n')
                break
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(f'\n[remote]\nbase_url = "{url}"\n')

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f'registered [remote] base_url = "{url}"')


def main(argv: list) -> int:
    if len(argv) == 2 and argv[0] == "--get":
        print(get(argv[1]))
        return 0
    if len(argv) == 3 and argv[0] == "--set":
        set_url(argv[1], argv[2])
        return 0
    sys.stderr.write("usage: server_url.py --get CONFIG | --set CONFIG URL\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
