"""Compare regenerated tables and figures with reference copies.

Every file under ``tables/`` and ``figures/`` of the reference directory is compared with the
file at the same relative path in the new directory. A file is

* **identical** if its bytes are identical;
* **equivalent** if it is a figure whose rendered content is the same although the bytes
  differ, as happens across platforms: a PDF whose decompressed drawing content is identical
  once negative zero is written as zero (some math libraries return -0.0 for marker
  coordinates), or a PNG preview with identical pixels (the zlib builds compress differently)
  or, when its PDF is identical or equivalent, with anti-aliasing differences of at most
  ``AA_TOLERANCE`` / 255 per channel (rasterizers differ slightly between CPU architectures);
* **different** otherwise: the first differing line (text), the share of differing pixels
  (PNG), or the byte count (other files).

Files present on only one side are reported as missing or extra.

Usage: ``python scripts/compare_outputs.py NEW REFERENCE``. Exits non-zero if any file is
different, missing, or extra.
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

SUBDIRS = ("tables", "figures")
TEXT_SUFFIXES = {".tex", ".json", ".csv", ".md", ".txt"}
STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
NEGATIVE_ZERO = re.compile(rb"(?<![\w.])-0(?![\w.])")
AA_TOLERANCE = 4  # largest channel difference (of 255) of a PNG preview whose PDF matches


@dataclass(frozen=True)
class Difference:
    path: str
    kind: str  # "missing", "extra", "differs"
    detail: str = ""


def files_under(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for sub in SUBDIRS
        if (root / sub).is_dir()
        for p in (root / sub).rglob("*")
        if p.is_file()
    }


def _text_detail(new: bytes, ref: bytes) -> str:
    a = new.decode("utf-8", errors="replace").splitlines()
    b = ref.decode("utf-8", errors="replace").splitlines()
    for i, (x, y) in enumerate(zip(a, b, strict=False), start=1):
        if x != y:
            return f"line {i}: {x[:80]!r} (reference: {y[:80]!r})"
    return f"{len(a)} lines (reference: {len(b)} lines)"


def _pixels(path: Path):
    import matplotlib.image as image

    return image.imread(path)


def _png_detail(new: Path, ref: Path, pdf_matches: bool) -> str | None:
    """None if the pixels are identical (or, when the PDF of the figure matches, differ by at
    most ``AA_TOLERANCE``), else a description of the difference."""
    import numpy as np

    a, b = _pixels(new), _pixels(ref)
    if a.shape != b.shape:
        return f"image size {a.shape[1]}x{a.shape[0]} (reference: {b.shape[1]}x{b.shape[0]})"
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64)) * 255
    if not delta.any() or (pdf_matches and delta.max() <= AA_TOLERANCE + 1e-9):
        return None
    pixels = np.any(delta > 0, axis=-1) if delta.ndim == 3 else delta > 0
    return f"{pixels.mean():.4%} of pixels differ, largest channel difference {delta.max():.0f}/255"


def _pdf_content(data: bytes) -> tuple[list[bytes], bytes]:
    """Decompressed streams with negative zero written as zero, and the object structure
    without stream lengths and file offsets."""
    streams = []
    for m in STREAM.finditer(data):
        raw = m.group(1)
        try:
            raw = zlib.decompressobj().decompress(raw)  # tolerates image streams without an end
        except zlib.error:
            pass
        streams.append(NEGATIVE_ZERO.sub(b"0", raw))
    skeleton = STREAM.sub(b"stream endstream", data)
    skeleton = re.sub(rb"/Length \d+(?!\d)(?! \d+ R)", b"/Length", skeleton)
    # Stream lengths written as separate objects ("/Length 42 0 R"): ignore their values.
    for number in set(re.findall(rb"/Length (\d+) 0 R", skeleton)):
        length_object = re.compile(rb"\n" + number + rb" 0 obj\n\d+\nendobj")
        skeleton = length_object.sub(b"\n" + number + b" 0 obj\nendobj", skeleton)
    skeleton = skeleton.split(b"\nxref\n")[0]
    return streams, skeleton


def compare(new: Path, reference: Path) -> tuple[list[str], list[Difference], list[str]]:
    """Identical files, differences, and equivalent figures, comparing ``new`` with
    ``reference``."""
    new_files, ref_files = files_under(new), files_under(reference)
    identical, equivalent, differences = [], [], []
    pdf_matches: set[str] = set()  # PDFs that are identical or equivalent (sorted before PNGs)
    for rel in sorted(ref_files - new_files):
        differences.append(Difference(rel, "missing"))
    for rel in sorted(new_files - ref_files):
        differences.append(Difference(rel, "extra"))
    for rel in sorted(new_files & ref_files):
        a, b = (new / rel).read_bytes(), (reference / rel).read_bytes()
        if a == b:
            identical.append(rel)
            if rel.endswith(".pdf"):
                pdf_matches.add(rel)
            continue
        suffix = Path(rel).suffix.lower()
        if suffix == ".png":
            detail = _png_detail(new / rel, reference / rel, rel[:-4] + ".pdf" in pdf_matches)
        elif suffix == ".pdf":
            detail = (
                None
                if _pdf_content(a) == _pdf_content(b)
                else (f"drawing content differs ({len(a)} bytes; reference: {len(b)} bytes)")
            )
        elif suffix in TEXT_SUFFIXES:
            detail = _text_detail(a, b)
        else:
            detail = f"{len(a)} bytes (reference: {len(b)} bytes)"
        if detail is None:
            equivalent.append(rel)
            if rel.endswith(".pdf"):
                pdf_matches.add(rel)
        else:
            differences.append(Difference(rel, "differs", detail))
    return identical, differences, equivalent


def report(
    identical: list[str],
    differences: list[Difference],
    reference: Path,
    equivalent: list[str] | None = None,
) -> str:
    equivalent = equivalent or []
    lines = [
        f"compared with {reference}: {len(identical)} identical, {len(equivalent)} equivalent, "
        f"{len(differences)} differ"
    ]
    if equivalent:
        lines.append(
            "  equivalent (same drawing content, different bytes): "
            + ", ".join(Path(p).name for p in equivalent)
        )
    for d in differences:
        lines.append(f"  {d.kind:8s} {d.path}" + (f": {d.detail}" if d.detail else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args(argv)
    identical, differences, equivalent = compare(args.new, args.reference)
    print(report(identical, differences, args.reference, equivalent))
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
