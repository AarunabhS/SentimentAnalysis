"""Restore the documented public input, verifying both upstream and converted hashes."""
from __future__ import annotations
import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch():
    target = ROOT / "data" / TARGET
    if target.exists():
        if sha256(target.read_bytes()) != TARGET_SHA256:
            raise ValueError("Existing input differs from the documented sample. It was left untouched; use --data with analysis.py for custom inputs.")
        print(f"Verified existing input: {target.relative_to(ROOT)}")
        return
    request = urllib.request.Request(URL, headers={"User-Agent": "portfolio-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    if sha256(content) != SOURCE_SHA256:
        raise ValueError("Upstream checksum mismatch; no input file was written. Review the publisher's version before updating the expected hash.")
    converted = convert(content)
    if sha256(converted) != TARGET_SHA256:
        raise ValueError("Converted checksum differs; no input file was written. Use the tested requirements.txt versions.")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an existing file, even if it appeared during the download.
    with target.open("xb") as stream:
        stream.write(converted)
    print(f"Downloaded and verified: {target.relative_to(ROOT)}")


TARGET = 'reviews.csv'
URL = 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'
SOURCE_SHA256 = 'afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343'
TARGET_SHA256 = 'd5a164daf0cfa59303f481dd528fe0b74ebca43dff30a30063597a82d6886428'

def convert(content):
    import csv
    archive = zipfile.ZipFile(io.BytesIO(content))
    rows = []
    for source, name in [("amazon", "amazon_cells_labelled.txt"), ("imdb", "imdb_labelled.txt"), ("yelp", "yelp_labelled.txt")]:
        text = archive.read("sentiment labelled sentences/" + name).decode("utf-8")
        parsed = [{"text": re.sub(r"\s+", " ", value).strip(), "label": int(label), "source": source}
                  for value, label in re.findall(r"(.*?)\t([01])(?:\r?\n|$)", text, re.S)]
        if len(parsed) != 1000:
            raise ValueError("Unexpected source row count: " + source)
        rows.extend(parsed)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["text", "label", "source"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")

if __name__ == "__main__":
    try:
        fetch()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Download error: {error}")
