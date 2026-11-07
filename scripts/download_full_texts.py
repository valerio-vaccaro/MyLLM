#!/usr/bin/env python3
"""Download and convert the complete public-domain/open-licensed training texts.

Outputs are written to ``data/raw``:

* ``bible_kjv_full.txt`` - the complete King James Bible from Project Gutenberg
* ``mastering_bitcoin_third_edition.txt`` - the complete third-edition source
  text, converted from AsciiDoc to readable plain text
* ``mastering_bitcoin/`` - one converted ``.txt`` file for every text source in
  the GitHub archive

The downloads are atomic and are validated before replacing an existing file.
Use ``--force`` to refresh files that are already present.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import tempfile
import os
from pathlib import Path

import requests
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
GUTENBERG_URLS = (
    "https://www.gutenberg.org/ebooks/10.txt.utf-8",
    "https://www.gutenberg.org/ebooks/30.txt.utf-8",
    # Ebook 30 is the current Gutenberg record explicitly marked complete.
    "https://www.gutenberg.org/cache/epub/30/pg30.txt",
    "https://www.gutenberg.org/files/30/30-0.txt",
    # Keep ebook 10 as a compatibility fallback for mirrors/older catalogs.
    "https://www.gutenberg.org/cache/epub/10/pg10.txt",
    "https://www.gutenberg.org/files/10/10-0.txt",
    # Plain-text mirrors used when Gutenberg's legacy download aliases are
    # unavailable from a network proxy.
    "https://raw.githubusercontent.com/michaelt/kjv/master/kjv.txt",
    "https://raw.githubusercontent.com/christislord12/KJV-Bible-Text/main/KJVBible.txt",
)
MASTERING_URL = (
    "https://github.com/bitcoinbook/bitcoinbook/archive/refs/tags/"
    "third_edition_print1.tar.gz"
)
WHITEPAPER_URLS = (
    "https://raw.githubusercontent.com/laanwj/bitcoin-whitepaper/master/bitcoin_whitepaper.txt",
    "https://raw.githubusercontent.com/bitcoinbook/bitcoinbook-spanish/master/bitcoin_whitepaper.txt",
)
TEXT_SUFFIXES = {
    ".adoc", ".asciidoc", ".asc", ".md", ".markdown", ".txt", ".html",
    ".css", ".json", ".xml", ".yml", ".yaml", ".csv", ".tex", "",
}
SKIP_NAMES = {".gitignore", ".gitattributes"}


def fetch(url: str, destination: Path, description: str) -> None:
    """Stream *url* into *destination* and replace it only after completion."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(20, 120),
            headers={"User-Agent": "myllm-training-data/1.0"},
            # Some managed environments intercept HTTPS with a local CA.  In
            # normal use certificate verification stays enabled; the opt-in
            # switch is useful there when the CA cannot be installed.
            verify=not os.environ.get("MYLLM_INSECURE_TLS"),
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with temporary.open("wb") as output:
                progress = tqdm(
                    total=total or None, unit="B", unit_scale=True,
                    desc=description, disable=not total,
                )
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        progress.update(len(chunk))
                progress.close()
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def valid_bible(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2_000_000:
        return False
    text = read_text(path).lower()
    return (
        "king james" in text
        and "genesis" in text
        and "revelation" in text
        and len(text.split()) > 600_000
    )


def download_bible(force: bool) -> Path:
    destination = RAW_DIR / "bible_kjv_full.txt"
    if not force and valid_bible(destination):
        print(f"✓ Bible already present: {destination.name}")
        return destination

    last_error = None
    for url in GUTENBERG_URLS:
        try:
            fetch(url, destination, "King James Bible")
            if valid_bible(destination):
                print(f"✓ Bible complete: {destination.stat().st_size / 1_048_576:.1f} MiB")
                return destination
            raise ValueError("Downloaded file does not appear to contain the complete Bible")
        except Exception as error:
            last_error = error
            print(f"  attempt failed ({url}): {error}")
    raise RuntimeError(f"Could not download complete KJV: {last_error}")


def asciidoc_to_text(source: str) -> str:
    """Make AsciiDoc readable while retaining prose, headings, and code."""
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    source = re.sub(r"^include::[^\n]+\n?", "", source, flags=re.MULTILINE)
    source = re.sub(r"^image::[^\n]+\n?", "", source, flags=re.MULTILINE)
    source = re.sub(r"^\[source[^\]]*\]\s*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"^\[[^\]]+\]\s*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"^:.*:\s*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"^//.*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"^\.{4,}\s*$", "", source, flags=re.MULTILINE)
    source = re.sub(r"^={1,6}\s+", "", source, flags=re.MULTILINE)
    source = re.sub(r"\[\[([^,\]]+)(?:,[^\]]*)?\]\]", r"\1", source)
    source = re.sub(r"<<[^,>]+,([^>]+)>>", r"\1", source)
    source = re.sub(r"\(\(\(.*?\)\)\)", "", source)
    source = re.sub(r"xref:[^\[]+\[([^\]]*)\]", r"\1", source)
    source = re.sub(r"link:([^\[]+)\[([^\]]*)\]", r"\2 (\1)", source)
    source = re.sub(r"https?://[^\s\[]+\[([^\]]*)\]", r"\1", source)
    source = re.sub(r"(?<!\w)([*_+#~])([^\n]+?)\1", r"\2", source)
    source = re.sub(r"^\s*[|*!]\s*", "", source, flags=re.MULTILINE)
    source = re.sub(r"\n{3,}", "\n\n", source)
    return source.strip() + "\n"


def archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = []
    for member in archive.getmembers():
        path = Path(member.name)
        if not member.isfile() or path.name in SKIP_NAMES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # Only the repository's files; this also prevents path traversal.
        if len(path.parts) < 2 or ".." in path.parts:
            continue
        members.append(member)
    return sorted(members, key=lambda item: item.name.lower())


def download_mastering_bitcoin(force: bool) -> Path:
    output_dir = RAW_DIR / "mastering_bitcoin"
    combined_path = RAW_DIR / "mastering_bitcoin_third_edition.txt"
    marker = output_dir / ".source_tag"
    source_is_current = marker.exists() and marker.read_text(encoding="utf-8").strip() == MASTERING_URL
    if not force and combined_path.is_file() and source_is_current:
        print(f"✓ Mastering Bitcoin already present: {combined_path.name}")
        return combined_path

    with tempfile.TemporaryDirectory(prefix="mastering-bitcoin-") as temporary_dir:
        archive_path = Path(temporary_dir) / "mastering-bitcoin.tar.gz"
        print("📥 Downloading Mastering Bitcoin archive (third_edition_print1)")
        fetch(MASTERING_URL, archive_path, "Mastering Bitcoin")
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive_members(archive)
            if not members:
                raise RuntimeError("archive contains no text sources")
            staging = Path(temporary_dir) / "converted"
            staging.mkdir()
            converted = []
            for member in members:
                source_file = archive.extractfile(member)
                if source_file is None:
                    continue
                text = source_file.read().decode("utf-8", errors="replace")
                relative = Path(*Path(member.name).parts[1:])
                target = staging / relative.with_suffix(".txt")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(asciidoc_to_text(text), encoding="utf-8")
                converted.append((relative, target))

            output_dir.mkdir(parents=True, exist_ok=True)
            for old_file in output_dir.rglob("*"):
                if old_file.is_file() and old_file.name != marker.name:
                    old_file.unlink()
            combined = [
                "MASTERING BITCOIN: PROGRAMMING THE OPEN BLOCKCHAIN\n",
                "Third Edition, first print (source archive: third_edition_print1)\n\n",
            ]
            for relative, source_file in converted:
                target = output_dir / relative.with_suffix(".txt")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_file, target)
                combined.append(f"\n\n{'=' * 80}\n{relative}\n{'=' * 80}\n\n")
                combined.append(source_file.read_text(encoding="utf-8"))
            combined_path.write_text("".join(combined), encoding="utf-8")
            marker.write_text(MASTERING_URL, encoding="utf-8")

    print(f"✓ Mastering Bitcoin converted: {len(converted)} files, {combined_path.stat().st_size / 1_048_576:.1f} MiB")
    return combined_path


def download_whitepaper(force: bool) -> Path | None:
    """Download Bitcoin Whitepaper from public sources."""
    destination = RAW_DIR / "bitcoin_whitepaper.txt"

    if not force and destination.is_file() and destination.stat().st_size > 10_000:
        print(f"✓ Bitcoin Whitepaper already present: {destination.name}")
        return destination

    last_error = None
    for url in WHITEPAPER_URLS:
        try:
            fetch(url, destination, "Bitcoin Whitepaper")
            text = read_text(destination)
            if len(text) > 10_000:  # Whitepaper should be ~30KB
                print(f"✓ Bitcoin Whitepaper complete: {destination.stat().st_size / 1_048_576:.2f} MiB")
                return destination
            raise ValueError("Downloaded file is too small")
        except Exception as error:
            last_error = error
            print(f"  attempt failed ({url}): {error}")

    print(f"⚠️ Could not download Bitcoin Whitepaper: {last_error}")
    return None


def report(paths: list[Path]) -> None:
    print("\n📊 Text corpus summary")
    for path in paths:
        if path:  # Skip None paths
            words = len(read_text(path).split())
            print(f"  {path.relative_to(BASE_DIR)}: {path.stat().st_size / 1_048_576:.1f} MiB, {words:,} words")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="redownload and recreate existing files")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    bible = download_bible(args.force)
    mastering = download_mastering_bitcoin(args.force)
    whitepaper = download_whitepaper(args.force)
    report([bible, mastering, whitepaper])
    print("\n✓ Corpus ready. Now run: python scripts/preprocess.py")


if __name__ == "__main__":
    main()
