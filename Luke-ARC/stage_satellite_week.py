"""Stage the June 2020 GSP 199 satellite week from OCF's public GCS Zarr.

Reads only the available required 5-minute scans and the 24x24 GSP crop.
Each batch is committed to a local checkpoint before proceeding, allowing a
rerun to reuse completed batches after a job interruption.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ocf_blosc2  # register the codec used by the public archive
import pandas as pd
import xarray as xr
import yaml
from numcodecs import Blosc
from ocf_data_sampler.load.satellite import open_sat_data
from ocf_data_sampler.select import Location
from ocf_data_sampler.torch_datasets.utils import (
    add_alterate_coordinate_projections,
    slice_datasets_by_space,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "Luke-ARC/configs/sampler_2020_smoke.yaml"
PUBLIC_SATELLITE_ZARR = (
    "gs://public-datasets-eumetsat-solar-forecasting/satellite/"
    "EUMETSAT/SEVIRI_RSS/v4/2020_nonhrv.zarr"
)
WEEK_START = pd.Timestamp("2020-06-21T00:00:00Z")
WEEK_END = pd.Timestamp("2020-06-28T00:00:00Z")  # exclusive
SAMPLE_INTERVAL = pd.Timedelta(minutes=30)
GSP_ID = 199
GSP_LOCATION = Location(x=-2.0111, y=51.3441, coord_system="lon_lat", id=GSP_ID)
IMAGE_SIZE = 24
CHANNEL_COUNT = 11
BATCH_SIZE = 60


def validate_part(part_store: Path, expected_times: pd.DatetimeIndex) -> bool:
    """Return False only for the known pre-fix (time, y, x, channel) layout."""
    part = xr.open_zarr(part_store, consolidated=True)
    try:
        part_times = pd.DatetimeIndex(pd.to_datetime(part.time.values))
        expected_shape = (len(expected_times), CHANNEL_COUNT, IMAGE_SIZE, IMAGE_SIZE)
        if not part_times.equals(expected_times):
            raise ValueError(
                f"Satellite checkpoint has unexpected timestamps: {part_store}. "
                "Inspect it before removing or replacing it."
            )
        variable = part["satellite_actual"]
        if variable.shape == expected_shape and tuple(variable.dims) == (
            "time", "variable", "x_geostationary", "y_geostationary"
        ):
            return True
        old_shape = (len(expected_times), IMAGE_SIZE, IMAGE_SIZE, CHANNEL_COUNT)
        old_channel_last = (
            variable.dims[0] == "time"
            and variable.dims[-1] == "variable"
            and set(variable.dims[1:-1]) == {"x_geostationary", "y_geostationary"}
        )
        if variable.shape == old_shape and old_channel_last:
            return False
        raise ValueError(
            f"Satellite checkpoint has an unrecognized layout: {part_store}; "
            f"got dims={variable.dims}, shape={variable.shape}."
        )
    finally:
        part.close()


def main() -> None:
    data_root_text = os.environ.get("PVNET_DATA_ROOT", "").strip()
    if not data_root_text:
        raise RuntimeError("Set PVNET_DATA_ROOT to the persistent ARC data directory before submission.")
    diagnostic_dir = Path(data_root_text).expanduser() / "public_bucket_diagnostic"
    output_store = diagnostic_dir / "satellite_gsp199_20200621_week_public_sampler.zarr"
    parts_dir = diagnostic_dir / "satellite_gsp199_20200621_week_parts"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    if output_store.exists():
        raise FileExistsError(f"Refusing to overwrite existing satellite sampler store: {output_store}")

    with DATA_CONFIG_PATH.open() as stream:
        data_cfg = yaml.safe_load(stream)["input_data"]
    satellite_cfg = dict(data_cfg["satellite"])
    channels = list(satellite_cfg["channels"])
    if len(channels) != CHANNEL_COUNT:
        raise ValueError(f"Expected {CHANNEL_COUNT} configured non-HRV channels, got {channels}")

    required_times = pd.date_range(
        WEEK_START.tz_localize(None) - pd.Timedelta(minutes=30),
        (WEEK_END - SAMPLE_INTERVAL).tz_localize(None),
        freq="5min",
    )
    if len(required_times) != 2017:
        raise ValueError(f"Expected 2,017 required scans, got {len(required_times)}")
    print(f"Opening public archive: {PUBLIC_SATELLITE_ZARR}", flush=True)
    print(f"Required time range: {required_times[0]} to {required_times[-1]} UTC", flush=True)

    public_ds = xr.open_zarr(
        PUBLIC_SATELLITE_ZARR,
        consolidated=False,
        chunks=None,
        storage_options={"token": "anon"},
    )
    try:
        data_vars = list(public_ds.data_vars)
        if len(data_vars) != 1:
            raise ValueError(f"Expected one satellite array in the public Zarr; found {data_vars}.")
        public_da = public_ds[data_vars[0]]
        rename = {}
        if "time" in public_da.dims:
            rename["time"] = "time_utc"
        if "variable" in public_da.dims:
            rename["variable"] = "channel"
        public_da = public_da.rename(rename)
        public_da = public_da.sortby("time_utc").sortby("x_geostationary").sortby("y_geostationary")
        public_da = public_da.sel(channel=channels)

        source_times = pd.DatetimeIndex(pd.to_datetime(public_da.time_utc.values))
        missing_times = required_times.difference(source_times)
        selected_times = required_times.intersection(source_times)
        if not len(selected_times):
            raise ValueError("The public archive contains none of the required scans.")
        if len(missing_times) != 8:
            raise ValueError(
                f"Archive coverage differs from the section 5a audit: expected 8 missing scans, "
                f"found {len(missing_times)}. Review timestamps before staging."
            )
        print(
            f"Archive coverage: {len(selected_times)} available / {len(required_times)} required; "
            f"{len(missing_times)} gaps preserved.",
            flush=True,
        )

        crop_cfg = dict(satellite_cfg)
        crop_cfg["zarr_path"] = str(output_store)
        crop_config = SimpleNamespace(
            input_data=SimpleNamespace(satellite=SimpleNamespace(**crop_cfg))
        )
        add_alterate_coordinate_projections([GSP_LOCATION], {"sat": public_da})
        spatial_da = slice_datasets_by_space({"sat": public_da}, GSP_LOCATION, crop_config)["sat"]
        got_grid_shape = (spatial_da.sizes["x_geostationary"], spatial_da.sizes["y_geostationary"])
        if got_grid_shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected a 24×24 GSP crop; got {got_grid_shape}.")

        reference_crop = diagnostic_dir / "20200621T1200Z_GSP199_8frame_context_sampler.zarr"
        if reference_crop.exists():
            reference = xr.open_zarr(reference_crop, consolidated=True)
            try:
                for coord in ("x_geostationary", "y_geostationary"):
                    if not np.allclose(spatial_da[coord].values, reference[coord].values, rtol=0, atol=1e-3):
                        raise ValueError(f"The staged {coord} grid differs from the validated GSP 199 crop.")
            finally:
                reference.close()

        area_attr = public_da.attrs.get("area")
        if area_attr is None:
            raise ValueError("The public satellite data has no area metadata.")
        batches = [
            selected_times[start:start + BATCH_SIZE]
            for start in range(0, len(selected_times), BATCH_SIZE)
        ]
        print(
            f"GSP {GSP_ID} crop: {IMAGE_SIZE}×{IMAGE_SIZE}; channels: {len(channels)}; "
            f"checkpoint batches: {len(batches)}",
            flush=True,
        )

        for batch_number, batch_times in enumerate(batches, start=1):
            part_store = parts_dir / f"batch_{batch_number:03d}.zarr"
            staging_store = Path(str(part_store) + ".staging")
            if part_store.exists():
                if validate_part(part_store, batch_times):
                    print(f"[{batch_number}/{len(batches)}] Reusing {part_store.name}", flush=True)
                    continue
                print(
                    f"[{batch_number}/{len(batches)}] Replacing checkpoint with the old channel-last layout: "
                    f"{part_store.name}",
                    flush=True,
                )
                shutil.rmtree(part_store)
            if staging_store.exists():
                print(f"[{batch_number}/{len(batches)}] Removing incomplete checkpoint {staging_store.name}", flush=True)
                shutil.rmtree(staging_store)

            batch_da = (
                spatial_da.sel(time_utc=batch_times.values)
                .transpose("time_utc", "channel", "x_geostationary", "y_geostationary")
                .load()
            )
            batch_ds = batch_da.rename({"time_utc": "time", "channel": "variable"}).to_dataset(
                name="satellite_actual"
            )
            batch_ds.attrs["area"] = area_attr
            try:
                batch_ds.to_zarr(
                    staging_store,
                    mode="w",
                    consolidated=True,
                    zarr_format=2,
                    encoding={
                        "satellite_actual": {
                            "compressor": Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
                            "chunks": (1, CHANNEL_COUNT, IMAGE_SIZE, IMAGE_SIZE),
                        }
                    },
                )
            finally:
                batch_ds.close()
            staging_store.rename(part_store)
            if not validate_part(part_store, batch_times):
                raise RuntimeError(f"Newly written checkpoint has the obsolete axis order: {part_store}")
            print(
                f"[{batch_number}/{len(batches)}] Saved {len(batch_times)} scans through "
                f"{batch_times[-1]} UTC",
                flush=True,
            )

        final_staging = Path(str(output_store) + ".staging")
        if final_staging.exists():
            print(f"Removing incomplete final staging store: {final_staging}", flush=True)
            shutil.rmtree(final_staging)
        parts = [
            xr.open_zarr(parts_dir / f"batch_{number:03d}.zarr", consolidated=True)
            for number in range(1, len(batches) + 1)
        ]
        try:
            combined = xr.concat(
                parts,
                dim="time",
                data_vars="minimal",
                coords="minimal",
                compat="equals",
                join="exact",
            )
            combined_times = pd.DatetimeIndex(pd.to_datetime(combined.time.values))
            if not combined_times.equals(selected_times):
                raise ValueError("Concatenated satellite checkpoints do not match the selected archive times.")
            combined.to_zarr(
                final_staging,
                mode="w",
                consolidated=True,
                zarr_format=2,
                encoding={
                    "satellite_actual": {
                        "compressor": Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
                        "chunks": (1, CHANNEL_COUNT, IMAGE_SIZE, IMAGE_SIZE),
                    }
                },
            )
        except Exception:
            if final_staging.exists():
                shutil.rmtree(final_staging)
            raise
        finally:
            for part in parts:
                part.close()
        final_staging.rename(output_store)
    finally:
        public_ds.close()

    staged = open_sat_data(str(output_store))
    try:
        staged_times = pd.DatetimeIndex(pd.to_datetime(staged.time_utc.values))
        expected_shape = (len(selected_times), CHANNEL_COUNT, IMAGE_SIZE, IMAGE_SIZE)
        if staged.shape != expected_shape or not staged_times.equals(selected_times):
            raise ValueError(f"Saved satellite Zarr failed validation: {staged.shape}; expected {expected_shape}.")
        print(f"Validated satellite sampler store: {staged.shape}", flush=True)
        print(f"Missing scans retained as gaps: {len(required_times) - len(staged_times)}", flush=True)
        print(f"Final store: {output_store}", flush=True)
    finally:
        staged.close()


if __name__ == "__main__":
    main()
