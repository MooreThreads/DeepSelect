#!/usr/bin/env python3
"""Retag a Linux wheel with a manylinux platform tag for PyPI upload.

PyPI rejects the generic ``linux_x86_64`` platform tag; prebuilt MUSA wheels are
built with ``linux_x86_64`` (they link against the MUSA runtime and are only
installable on glibc >= the manylinux baseline).  This script rewrites only the
wheel filename + ``WHEEL`` metadata tag and recomputes ``RECORD``; file contents
(including the compiled extension) are unchanged.

Usage:
    python3 retag_manylinux.py \
        --wheel dist/deepselect_musa-0.0.1-cp310-cp310-linux_x86_64.whl \
        --platform-tag manylinux_2_35_x86_64 \
        [--remove]

The script is intentionally dependency-free (stdlib only) so it works in the
GitHub-hosted publish job without extra wheels.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import pathlib
import re
import sys
import zipfile

WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(?:-(?P<build>\d[^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$"
)


def _urlsafe_b64_digest(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def retag(wheel: pathlib.Path, platform_tag: str, remove_input: bool) -> pathlib.Path:
    match = WHEEL_RE.match(wheel.name)
    if not match:
        raise SystemExit(f"not a valid wheel filename: {wheel.name}")
    parts = match.groupdict()
    if parts["plat"] == platform_tag:
        print(f"wheel already tagged {platform_tag}: {wheel}", file=sys.stderr)
        return wheel
    new_name = (
        f"{parts['name']}-{parts['version']}"
        + (f"-{parts['build']}" if parts["build"] else "")
        + f"-{parts['py']}-{parts['abi']}-{platform_tag}.whl"
    )
    out = wheel.with_name(new_name)

    with zipfile.ZipFile(wheel) as zin:
        infos = zin.infolist()
        payload = {info.filename: zin.read(info.filename) for info in infos}

    dist_info = next(
        (n for n in payload if n.endswith(".dist-info/WHEEL")), None
    )
    if dist_info is None:
        raise SystemExit(f"no .dist-info/WHEEL in {wheel}")

    wheel_text = payload[dist_info].decode("utf-8")
    old_tag = f"{parts['py']}-{parts['abi']}-{parts['plat']}"
    new_tag = f"{parts['py']}-{parts['abi']}-{platform_tag}"
    if old_tag not in wheel_text:
        raise SystemExit(
            f"WHEEL metadata does not contain tag {old_tag}; refusing to retag"
        )
    payload[dist_info] = wheel_text.replace(old_tag, new_tag).encode("utf-8")

    record_name = dist_info[: -len("WHEEL")] + "RECORD"
    record_rows = []
    for info in infos:
        name = info.filename
        if name == record_name:
            continue
        data = payload[name]
        record_rows.append([name, f"sha256={_urlsafe_b64_digest(data)}", str(len(data))])
    record_rows.append([record_name, "", ""])
    record_buf = io.StringIO()
    csv.writer(record_buf, lineterminator="\n").writerows(record_rows)
    payload[record_name] = record_buf.getvalue().encode("utf-8")

    with zipfile.ZipFile(out, "w") as zout:
        for info in infos:
            data = payload[info.filename]
            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            new_info.create_system = info.create_system
            new_info.comment = info.comment
            zout.writestr(new_info, data)

    if remove_input:
        wheel.unlink()
    print(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=pathlib.Path)
    parser.add_argument(
        "--platform-tag",
        default="manylinux_2_35_x86_64",
        help="target platform tag (default: manylinux_2_35_x86_64)",
    )
    parser.add_argument("--remove", action="store_true", help="remove the input wheel")
    args = parser.parse_args()
    if not args.wheel.is_file():
        parser.error(f"no such wheel: {args.wheel}")
    retag(args.wheel, args.platform_tag, args.remove)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
