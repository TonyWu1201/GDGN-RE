"""Download a public source without replacing existing files; retain provenance."""

import argparse
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
import urllib.parse
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, source_id, release_id="unverified"):
    destination = Path(destination)
    root = Path(__file__).resolve().parents[2]
    destination = destination.resolve()
    destination.relative_to(root / "data")
    destination.parent.mkdir(parents=True, exist_ok=True)
    evidence = root / "data/manifests/acquisition"
    evidence.mkdir(parents=True, exist_ok=True)
    record = {"source_id": source_id, "release_id": release_id, "download_url": url,
              "local_path": destination.relative_to(root).as_posix(),
              "accessed_at": datetime.now(timezone.utc).isoformat()}
    record_path = evidence / (destination.name + ".json")
    if destination.exists():
        print(f"Exists; left unchanged: {destination.name}", flush=True)
        return
    part = destination.with_name(destination.name + ".part")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "GDGN-rebuild-data-acquisition/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                resolved = urllib.parse.urlsplit(response.url)
                stable_url = urllib.parse.urlunsplit((resolved.scheme, resolved.netloc, resolved.path, "", ""))
                record.update(http_status=response.status, final_url=stable_url,
                              content_type=response.headers.get("Content-Type"),
                              content_length=response.headers.get("Content-Length"),
                              etag=response.headers.get("ETag"),
                              last_modified=response.headers.get("Last-Modified"))
                if "text/html" in (record["content_type"] or "") and destination.suffix != ".html":
                    raise ValueError("HTML received instead of data (possibly verification page)")
                with part.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if record["content_length"] and part.stat().st_size != int(record["content_length"]):
                raise ValueError("Content-Length mismatch")
            if destination.suffix == ".gz":
                with gzip.open(part, "rb") as stream:
                    while stream.read(8 * 1024 * 1024):
                        pass
            elif destination.suffix in {".zip", ".xlsx"}:
                with zipfile.ZipFile(part) as archive:
                    if archive.testzip() is not None:
                        raise ValueError("ZIP CRC verification failed")
            elif destination.suffix == ".pdf":
                with part.open("rb") as stream:
                    if stream.read(5) != b"%PDF-":
                        raise ValueError("Invalid PDF signature")
            record.update(sha256=sha256(part), file_size=part.stat().st_size,
                          downloaded_at=datetime.now(timezone.utc).isoformat(),
                          acquisition_status="downloaded_verified_container")
            part.rename(destination)
            break
        except Exception as exc:
            record.update(acquisition_status="failed", error=f"{type(exc).__name__}: {exc}", attempt=attempt)
            print(f"{destination.name}: {record['error']}", flush=True)
            if isinstance(exc, urllib.error.HTTPError) and exc.code in {401, 403, 404, 410}:
                break
            if isinstance(exc, ValueError):
                break
            if attempt < 3:
                time.sleep(2 * attempt)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--release-id", default="unverified")
    args = parser.parse_args()
    result = download(args.url, args.destination, args.source_id, args.release_id)
    if result and result["acquisition_status"] == "failed":
        raise SystemExit(1)
