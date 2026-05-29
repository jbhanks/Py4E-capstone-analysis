from pathlib import Path
import faulthandler
import hashlib
import os
import resource
import sys
import time
import urllib.request
import urllib.error


faulthandler.enable()


ABOUT_URL = "https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5/about_data"

CSV_URL = (
    "https://data.cityofnewyork.us/api/views/wvxf-dwi5/rows.csv"
    "?accessType=DOWNLOAD"
)

OUT_DIR = Path("data/probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def download(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    print(f"\nPID: {os.getpid()}", flush=True)
    print(f"URL: {url}", flush=True)
    print(f"OUT: {out_path}", flush=True)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "python-download-probe/1.0",
            "Accept": "*/*",
        },
    )

    started = time.monotonic()
    total = 0
    digest = hashlib.sha256()

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            print(f"status: {response.status}", flush=True)
            print(f"content-type: {response.headers.get('content-type')}", flush=True)
            print(f"content-length: {response.headers.get('content-length')}", flush=True)

            with out_path.open("wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)

                    if total % (100 * 1024 * 1024) < chunk_size:
                        elapsed = time.monotonic() - started
                        mib = total / 1024 / 1024
                        speed = mib / elapsed if elapsed else 0
                        print(
                            f"{mib:,.1f} MiB downloaded | "
                            f"{speed:,.1f} MiB/s | "
                            f"max RSS {rss_mib():,.1f} MiB",
                            flush=True,
                        )

    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code} {e.reason}", file=sys.stderr, flush=True)
        print(e.read(500).decode("utf-8", errors="replace"), file=sys.stderr, flush=True)
        raise

    elapsed = time.monotonic() - started
    print("\nfinished", flush=True)
    print(f"bytes: {total:,}", flush=True)
    print(f"MiB: {total / 1024 / 1024:,.1f}", flush=True)
    print(f"seconds: {elapsed:,.1f}", flush=True)
    print(f"sha256: {digest.hexdigest()}", flush=True)
    print(f"max RSS: {rss_mib():,.1f} MiB", flush=True)


def main() -> None:
    print(f"python: {sys.executable}", flush=True)

    print("\n=== Testing about_data page ===", flush=True)
    download(ABOUT_URL, OUT_DIR / "about_data.html")

    print("\n=== Testing CSV export ===", flush=True)
    download(CSV_URL, OUT_DIR / "hmc_violations.csv")


if __name__ == "__main__":
    main()

