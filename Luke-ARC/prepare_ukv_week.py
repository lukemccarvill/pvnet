"""Download, crop, and assemble the June 2020 GSP 199 UKV week on HTC.

The job is restartable at cycle boundaries. Completed per-cycle crops are kept
under ``$PVNET_DATA_ROOT/public_bucket_diagnostic/ukv_gsp199_20200621_week_cycles``;
raw GRIBs are deleted only after their crop has been saved successfully.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import requests
import xarray as xr
import yaml
from eccodes import codes_get, codes_grib_new_from_file, codes_release


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "Luke-ARC/configs/sampler_2020_smoke.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "Luke-ARC/configs/late_fusion_3h_smoke.yaml"
WEEK_START = pd.Timestamp("2020-06-21T00:00:00Z")
WEEK_END = pd.Timestamp("2020-06-28T00:00:00Z")  # exclusive
SAMPLE_INTERVAL = pd.Timedelta(minutes=30)
GSP_ID = 199
GSP_NAME = "MELK_1"
CYCLE_HOURS = 3
WHOLESALES = (1, 2)
CHANNEL_SOURCE = {
    "t": "t", "dswrf": "sdswrf", "dlwrf": "sdlwrf", "hcc": "hcc",
    "mcc": "mcc", "lcc": "lcc", "sde": "sde", "r": "r", "vis": "vis",
    "si10": "si10", "wdir10": "wdir10", "prate": "prate", "hcct": "hcct",
    "cdcb": "cbh", "dpt": "dpt", "prmsl": "prmsl", "h": "h",
}


def retry_delay(headers: dict[str, str], attempt: int) -> float:
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                return max(1.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return min(300.0, 15.0 * (2 ** (attempt - 1)))


def download_file(url: str, destination: Path, token: str, max_attempts: int = 10) -> tuple[int, str]:
    """Stream a GRIB to disk, resuming partial bytes and backing off on 429/5xx."""
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file() and destination.stat().st_size:
        return destination.stat().st_size, "already present"

    for attempt in range(1, max_attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Authorization": f"Bearer {token}"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            should_retry = False
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt == max_attempts:
                        response.raise_for_status()
                    delay = retry_delay(response.headers, attempt)
                    print(
                        f"CEDA HTTP {response.status_code} for {destination.name}; "
                        f"Retry-After={response.headers.get('Retry-After')!r}; "
                        f"retry {attempt}/{max_attempts} in {delay:.0f}s.",
                        flush=True,
                    )
                    should_retry = True
                else:
                    response.raise_for_status()
                    if "text/html" in response.headers.get("Content-Type", "").lower():
                        raise RuntimeError(f"CEDA returned HTML instead of GRIB for {destination.name}")
                    append = offset > 0 and response.status_code == 206
                    if offset and not append:
                        offset = 0
                    content_range = response.headers.get("Content-Range", "")
                    expected_total = (
                        int(content_range.rsplit("/", 1)[1]) if "/" in content_range
                        else int(response.headers["Content-Length"]) + offset
                        if response.headers.get("Content-Length") else None
                    )
                    with partial.open("ab" if append else "wb") as output:
                        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                output.write(chunk)
                    actual_size = partial.stat().st_size
                    if expected_total is not None and actual_size != expected_total:
                        raise IOError(
                            f"Incomplete {destination.name}: got {actual_size:,} of "
                            f"{expected_total:,} bytes; partial retained for resume."
                        )
                    partial.replace(destination)
                    return actual_size, "downloaded"
            if should_retry:
                time.sleep(delay)
                continue
        except requests.RequestException as error:
            status = error.response.status_code if isinstance(error, requests.HTTPError) and error.response else None
            if status is not None and status != 429 and not 500 <= status < 600:
                raise
            if attempt == max_attempts:
                raise
            delay = min(300.0, 15.0 * (2 ** (attempt - 1)))
            print(f"Request failed for {destination.name}: {error}; retry in {delay:.0f}s.", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Failed to download {destination.name} after {max_attempts} attempts.")


def read_grid_metadata(grib_path: Path) -> dict[str, int | float]:
    with grib_path.open("rb") as stream:
        while True:
            gid = codes_grib_new_from_file(stream)
            if gid is None:
                break
            try:
                if codes_get(gid, "shortName") == "t" and codes_get(gid, "level") == 1:
                    return {
                        "nx": int(codes_get(gid, "Ni")), "ny": int(codes_get(gid, "Nj")),
                        "x_first": float(codes_get(gid, "X1InGridLengths")),
                        "x_last": float(codes_get(gid, "X2InGridLengths")),
                        "y_first": float(codes_get(gid, "Y1InGridLengths")),
                        "y_last": float(codes_get(gid, "Y2InGridLengths")),
                    }
            finally:
                codes_release(gid)
    raise ValueError(f"Could not read UKV grid metadata from {grib_path}")


def load_configuration():
    with DATA_CONFIG_PATH.open() as stream:
        data_cfg = yaml.safe_load(stream)["input_data"]
    with MODEL_CONFIG_PATH.open() as stream:
        model_cfg = yaml.safe_load(stream)["model"]
    ukv_cfg = data_cfg["nwp"]["ukv"]
    if list(ukv_cfg["channels"]) != list(CHANNEL_SOURCE):
        raise ValueError("UKV channel list differs from the mapping used by this worker.")
    return data_cfg, model_cfg, ukv_cfg


def main() -> None:
    data_root_text = os.environ.get("PVNET_DATA_ROOT", "").strip()
    if not data_root_text:
        raise RuntimeError("Set PVNET_DATA_ROOT to the persistent ARC data directory before submission.")
    token = os.environ.get("CEDA_TOKEN", "").strip()
    if not token:
        raise RuntimeError("CEDA_TOKEN was not passed to this batch job. See Luke-ARC/README.md.")

    data_root = Path(data_root_text).expanduser()
    diagnostic_dir = data_root / "public_bucket_diagnostic"
    pilot_dir = diagnostic_dir / "ukv_20200621_06"
    raw_dir = diagnostic_dir / "ukv_20200621_28_raw"
    crop_dir = diagnostic_dir / "ukv_gsp199_20200621_week_cycles"
    zarr_path = diagnostic_dir / "ukv_gsp199_20200621_week_sampler.zarr"
    for path in (raw_dir, crop_dir):
        path.mkdir(parents=True, exist_ok=True)
    if zarr_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing sampler store: {zarr_path}")

    data_cfg, model_cfg, ukv_cfg = load_configuration()
    sample_t0s = pd.date_range(WEEK_START, WEEK_END - SAMPLE_INTERVAL, freq=SAMPLE_INTERVAL)
    if len(sample_t0s) != 336:
        raise ValueError(f"Expected 336 half-hour sample times; found {len(sample_t0s)}")
    nwp_step_minutes = int(ukv_cfg["time_resolution_minutes"])
    nwp_length = (
        int(model_cfg["nwp_history_minutes"]["ukv"])
        + int(model_cfg["nwp_forecast_minutes"]["ukv"])
    ) // nwp_step_minutes + 1
    dropout_minutes = min(map(int, ukv_cfg["dropout_timedeltas_minutes"]))
    available_at_t0 = sample_t0s + pd.Timedelta(minutes=dropout_minutes)
    required_inits = pd.DatetimeIndex(
        available_at_t0.floor(f"{CYCLE_HOURS}h").unique()
    ).sort_values()

    pilot_grib = pilot_dir / "202006210600_u1096_ng_umqv_Wholesale1.grib"
    if not pilot_grib.is_file():
        raise FileNotFoundError(f"Missing verified pilot GRIB: {pilot_grib}")
    grid_info = read_grid_metadata(pilot_grib)
    locations_path = (
        Path(__import__("ocf_data_sampler").__file__).parent
        / "gsp_location_data_legacy_format" / "uk_gsp_locations_20250109.csv"
    )
    locations = pd.read_csv(locations_path, usecols=["gsp_id", "gsp_name", "x_osgb", "y_osgb"])
    location = locations.loc[locations.gsp_id.eq(GSP_ID)]
    if len(location) != 1 or location.iloc[0].gsp_name != GSP_NAME:
        raise LookupError(f"Could not resolve GSP {GSP_ID} ({GSP_NAME}) from OCF location data.")
    gsp_x, gsp_y = float(location.iloc[0].x_osgb), float(location.iloc[0].y_osgb)
    ukv_x = np.linspace(grid_info["x_first"], grid_info["x_last"], grid_info["nx"])
    ukv_y = np.linspace(grid_info["y_first"], grid_info["y_last"], grid_info["ny"])[::-1]
    center_i = int(np.abs(ukv_x - gsp_x).argmin())
    center_j = int(np.abs(ukv_y - gsp_y).argmin())
    x_slice = slice(center_i - 12, center_i + 12)
    y_slice = slice(center_j - 12, center_j + 12)
    if x_slice.start < 0 or y_slice.start < 0 or x_slice.stop > len(ukv_x) or y_slice.stop > len(ukv_y):
        raise ValueError("The GSP 199 24x24 crop extends beyond the UKV grid.")
    if not np.allclose(np.diff(ukv_x), 2000) or not np.allclose(np.diff(ukv_y), 2000):
        raise ValueError("Expected 2 km UKV grid spacing.")

    print(f"Week: {WEEK_START} to {WEEK_END} UTC (end exclusive)", flush=True)
    print(f"UKV cycles: {len(required_inits)}; GRIBs: {len(required_inits) * len(WHOLESALES)}", flush=True)
    print(f"Raw GRIB directory: {raw_dir}", flush=True)
    print(f"Per-cycle crop directory: {crop_dir}", flush=True)
    print(f"Final sampler store: {zarr_path}", flush=True)

    crop_paths = []
    for cycle_number, init_time in enumerate(required_inits, start=1):
        crop_path = crop_dir / f"ukv_gsp199_{init_time:%Y%m%dT%H%MZ}.nc"
        crop_paths.append(crop_path)
        if crop_path.is_file() and crop_path.stat().st_size > 0:
            print(f"[{cycle_number}/{len(required_inits)}] Reusing {crop_path.name}", flush=True)
            continue

        rows = []
        for wholesale in WHOLESALES:
            filename = f"{init_time:%Y%m%d%H%M}_u1096_ng_umqv_Wholesale{wholesale}.grib"
            url = (
                "https://dap.ceda.ac.uk/badc/ukmo-nwp/data/ukv-grib/"
                f"{init_time:%Y/%m/%d}/{filename}?download=1"
            )
            rows.append((filename, url))
        cycle_paths = []
        for filename, url in rows:
            pilot_or_manual = pilot_dir / filename
            raw_path = raw_dir / filename
            if pilot_or_manual.is_file() and pilot_or_manual.stat().st_size > 0:
                cycle_paths.append(pilot_or_manual)
                print(f"  Reusing existing GRIB {filename}", flush=True)
            else:
                size, state = download_file(url, raw_path, token)
                print(f"  {state}: {filename} ({size / 1e6:.1f} MB)", flush=True)
                time.sleep(3)
                cycle_paths.append(raw_path)

        # Determine the valid-time fields needed by samples assigned to this init.
        assigned = sample_t0s[available_at_t0.floor(f"{CYCLE_HOURS}h") == init_time]
        target_times = set()
        for t0 in assigned:
            start = (t0 + pd.Timedelta(minutes=ukv_cfg["interval_start_minutes"])).ceil(f"{nwp_step_minutes}min")
            target_times.update(pd.date_range(start, start + (nwp_length - 1) * pd.Timedelta(minutes=nwp_step_minutes), freq=f"{nwp_step_minutes}min"))
        target_times = pd.DatetimeIndex(sorted(target_times))
        steps = np.asarray([(valid_time - init_time).to_timedelta64() for valid_time in target_times])
        if np.any(steps < np.timedelta64(0, "h")) or np.any(steps > np.timedelta64(36, "h")):
            raise ValueError(f"Required UKV leads outside 0–36 h for {init_time}: {steps}")

        current_grid = read_grid_metadata(cycle_paths[0])
        if current_grid != grid_info:
            raise ValueError(f"UKV grid changed at {init_time}: {current_grid}")
        values_by_channel = {}
        for grib_path in cycle_paths:
            with xr.set_options(use_new_combine_kwarg_defaults=True):
                groups = cfgrib.open_datasets(str(grib_path), backend_kwargs={"indexpath": ""})
            try:
                for ds in groups:
                    for channel, source_var in CHANNEL_SOURCE.items():
                        if channel in values_by_channel or source_var not in ds.data_vars:
                            continue
                        field = ds[source_var]
                        if channel == "t":
                            field = field.sel(heightAboveGround=1)
                        field = field.sel(step=steps)
                        values = np.asarray(field.values, dtype=np.float32)
                        expected = (len(steps), grid_info["nx"] * grid_info["ny"])
                        if values.shape != expected:
                            raise ValueError(f"Unexpected {source_var} shape in {grib_path.name}: {values.shape}; expected {expected}")
                        values = values.reshape(len(steps), grid_info["ny"], grid_info["nx"])[:, ::-1, :]
                        values_by_channel[channel] = values[:, y_slice, x_slice]
            finally:
                for ds in groups:
                    ds.close()
        missing_channels = [channel for channel in CHANNEL_SOURCE if channel not in values_by_channel]
        if missing_channels:
            raise ValueError(f"Missing UKV channels at {init_time}: {missing_channels}")

        cycle_values = np.stack([values_by_channel[c] for c in CHANNEL_SOURCE], axis=1)
        hcct_index = list(CHANNEL_SOURCE).index("hcct")
        cycle_values[:, hcct_index] = np.where(cycle_values[:, hcct_index] < 0, np.nan, cycle_values[:, hcct_index])
        cycle_values = np.transpose(cycle_values, (0, 1, 3, 2))
        cycle_ds = xr.Dataset(
            {"UKV": (("init_time", "step", "variable", "x", "y"), cycle_values[None, ...])},
            coords={
                "init_time": [init_time.tz_convert(None).to_datetime64()], "step": steps,
                "variable": list(CHANNEL_SOURCE), "x": ukv_x[x_slice], "y": ukv_y[y_slice],
            },
        )
        temporary_crop = crop_path.with_suffix(".nc.part")
        cycle_ds.to_netcdf(
            temporary_crop, engine="h5netcdf",
            encoding={"UKV": {"zlib": True, "complevel": 2, "_FillValue": -9999.0}},
        )
        temporary_crop.replace(crop_path)
        step_count = cycle_ds.sizes["step"]
        cycle_ds.close()
        for path in cycle_paths:
            if path.parent == raw_dir:
                path.unlink(missing_ok=True)
        print(f"[{cycle_number}/{len(required_inits)}] Saved {crop_path.name}; {step_count} valid times", flush=True)

    missing_crops = [path for path in crop_paths if not path.is_file() or not path.stat().st_size]
    if missing_crops:
        raise FileNotFoundError(f"Missing cycle crops: {missing_crops[:3]}")
    temp_zarr = zarr_path.with_name(zarr_path.name + ".building")
    if temp_zarr.exists():
        print(f"Removing incomplete temporary Zarr; all cycle crops are checkpointed: {temp_zarr}", flush=True)
        shutil.rmtree(temp_zarr)
    cycle_datasets = [xr.open_dataset(path, engine="h5netcdf") for path in crop_paths]
    try:
        week_ukv = xr.concat(cycle_datasets, dim="init_time").sortby("init_time")
        week_ukv.to_zarr(
            temp_zarr, mode="w", consolidated=True, zarr_format=2,
            encoding={"UKV": {"_FillValue": np.nan}},
        )
    except Exception:
        if temp_zarr.exists():
            shutil.rmtree(temp_zarr)
        raise
    finally:
        for dataset in cycle_datasets:
            dataset.close()
    temp_zarr.rename(zarr_path)
    print(f"Completed UKV sampler store: {zarr_path}", flush=True)
    print(f"Cycles/crops: {len(crop_paths)}; variables: {len(CHANNEL_SOURCE)}; crop: 24x24", flush=True)


if __name__ == "__main__":
    main()
