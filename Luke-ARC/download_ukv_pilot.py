"""Download the small UKV GRIB needed to identify the GSP crop grid on ARC."""

from __future__ import annotations

import os
import time
from getpass import getpass
from pathlib import Path

import requests


FILENAME = "202006210600_u1096_ng_umqv_Wholesale1.grib"
URL = f"https://dap.ceda.ac.uk/badc/ukmo-nwp/data/ukv-grib/2020/06/21/{FILENAME}?download=1"


def main() -> None:
    configured_data_root = os.environ.get("PVNET_DATA_ROOT")
    if not configured_data_root:
        raise RuntimeError("Set PVNET_DATA_ROOT to a persistent ARC $DATA directory first.")
    diagnostic_dir = Path(configured_data_root).expanduser() / "public_bucket_diagnostic"
    output_dir = diagnostic_dir / "ukv_20200621_06"
    destination = output_dir / FILENAME
    partial = destination.with_suffix(destination.suffix + ".part")
    output_dir.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Pilot GRIB already exists ({destination.stat().st_size / 1e6:.1f} MB): {destination}")
        return

    token = os.environ.get("CEDA_TOKEN", "").strip() or getpass("CEDA bearer token (hidden): ").strip()
    if not token:
        raise RuntimeError("No CEDA token was supplied.")

    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Authorization": f"Bearer {token}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=(30, 180)) as response:
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt == 8:
                        response.raise_for_status()
                    delay = min(300, 15 * 2 ** (attempt - 1))
                    print(f"CEDA returned HTTP {response.status_code}; retrying in {delay}s ({attempt}/8).")
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("CEDA returned HTML instead of a GRIB file; check the token/access.")

                append = offset > 0 and response.status_code == 206
                if offset and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                content_range = response.headers.get("Content-Range", "")
                if "/" in content_range:
                    expected_size = int(content_range.rsplit("/", 1)[1])
                elif response.headers.get("Content-Length"):
                    expected_size = int(response.headers["Content-Length"]) + offset
                else:
                    expected_size = None
                with partial.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            output.write(chunk)
                actual_size = partial.stat().st_size
                if expected_size is not None and actual_size != expected_size:
                    raise IOError(f"Incomplete transfer: received {actual_size:,} of {expected_size:,} bytes.")
                partial.replace(destination)
                print(f"Downloaded {actual_size / 1e6:.1f} MB to {destination}")
                return
        except (requests.RequestException, IOError) as exc:
            status = exc.response.status_code if isinstance(exc, requests.HTTPError) and exc.response is not None else None
            if status is not None and status != 429 and not 500 <= status < 600:
                raise
            if attempt == 8:
                raise
            delay = min(300, 15 * 2 ** (attempt - 1))
            print(f"Request failed ({exc}); retrying in {delay}s ({attempt}/8).")
            time.sleep(delay)

    raise RuntimeError(f"Could not download {FILENAME} after eight attempts.")


if __name__ == "__main__":
    main()
