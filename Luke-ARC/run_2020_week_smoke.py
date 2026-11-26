"""Validate and train the one-week PVNet 2020 smoke test on HTC."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ocf_blosc2
import pandas as pd
import torch
import yaml
from hydra.utils import instantiate
from lightning.pytorch import Callback, Trainer
from ocf_data_sampler.load.satellite import open_sat_data
from ocf_data_sampler.select import fill_time_periods, find_contiguous_t0_periods
from omegaconf import OmegaConf
from pvnet.datamodule import PVNetDataModule
from pvnet.utils import validate_batch_against_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG_PATH = REPO_ROOT / "Luke-ARC/configs/sampler_2020_smoke.yaml"
MODEL_CONFIG_PATH = REPO_ROOT / "Luke-ARC/configs/late_fusion_3h_smoke.yaml"
WEEK_START = pd.Timestamp("2020-06-21T00:00:00Z")
WEEK_END = pd.Timestamp("2020-06-28T00:00:00Z")
SAMPLE_INTERVAL = pd.Timedelta(minutes=30)
SAT_INTERVAL_START = pd.Timedelta(minutes=-30)
FORECAST_EMBARGO = pd.Timedelta(minutes=180)
GSP_ID = 199


class MetricHistory(Callback):
    def __init__(self) -> None:
        self.rows: list[dict[str, float | int]] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        row: dict[str, float | int] = {"epoch": trainer.current_epoch + 1}
        for name, value in trainer.callback_metrics.items():
            if torch.is_tensor(value) and value.numel() == 1:
                row[name] = float(value.detach().cpu())
        self.rows.append(row)


def main() -> None:
    data_root_text = os.environ.get("PVNET_DATA_ROOT", "").strip()
    if not data_root_text:
        raise RuntimeError("Set PVNET_DATA_ROOT to the persistent ARC data directory before submission.")
    diagnostic_dir = Path(data_root_text).expanduser() / "public_bucket_diagnostic"
    satellite_store = diagnostic_dir / "satellite_gsp199_20200621_week_public_sampler.zarr"
    ukv_store = diagnostic_dir / "ukv_gsp199_20200621_week_sampler.zarr"
    pvlive_store = diagnostic_dir / "pvlive_gsp199_20200621_week_sampler_backup_checked.zarr"
    required_stores = (satellite_store, ukv_store, pvlive_store)
    missing = [str(path) for path in required_stores if not path.exists()]
    if missing:
        raise FileNotFoundError("Required week stores are missing:\n" + "\n".join(missing))

    # Reproduce Section 5a's valid-time and chronological split from the saved
    # satellite coordinate, without relying on the NX-hosted notebook kernel.
    satellite = open_sat_data(str(satellite_store))
    try:
        available_times = pd.DatetimeIndex(pd.to_datetime(satellite.time_utc.values))
        if satellite.shape != (2009, 11, 24, 24):
            raise ValueError(f"Unexpected staged satellite shape: {satellite.shape}")
    finally:
        satellite.close()
    nominal_t0s = pd.date_range(WEEK_START.tz_localize(None), (WEEK_END - SAMPLE_INTERVAL).tz_localize(None), freq=SAMPLE_INTERVAL)
    continuity_periods = find_contiguous_t0_periods(
        available_times,
        interval_start=SAT_INTERVAL_START,
        interval_end=pd.Timedelta(0),
        time_resolution=pd.Timedelta(minutes=5),
    )
    sampler_valid_t0s = pd.DatetimeIndex(fill_time_periods(continuity_periods, freq=SAMPLE_INTERVAL))
    valid_t0s = nominal_t0s.intersection(sampler_valid_t0s)
    dropped_t0s = nominal_t0s.difference(valid_t0s)
    split_index = round(0.8 * len(valid_t0s))
    candidate_train = valid_t0s[:split_index]
    validation_t0s = valid_t0s[split_index:]
    if not len(candidate_train) or not len(validation_t0s):
        raise ValueError("Satellite coverage leaves an empty train or validation split.")
    train_t0s = candidate_train[candidate_train + FORECAST_EMBARGO < validation_t0s[0]]
    purged_t0s = candidate_train.difference(train_t0s)
    if not len(train_t0s) or train_t0s[-1] + FORECAST_EMBARGO >= validation_t0s[0]:
        raise ValueError("Training forecast horizon overlaps the validation period.")

    expected_split = (259, 66, 8, 6)
    observed_split = (len(train_t0s), len(validation_t0s), len(dropped_t0s), len(purged_t0s))
    if observed_split != expected_split:
        raise ValueError(f"Coverage/split differs from audited result: got {observed_split}, expected {expected_split}.")

    with DATA_CONFIG_PATH.open() as stream:
        data_config = yaml.safe_load(stream)
    week_input = data_config["input_data"]
    week_input["nwp"] = {"ukv": week_input["nwp"]["ukv"]}
    week_input["satellite"]["zarr_path"] = str(satellite_store.resolve())
    week_input["generation"]["zarr_path"] = str(pvlive_store.resolve())
    week_input["generation"].update(
        interval_start_minutes=-120, interval_end_minutes=180, time_resolution_minutes=30
    )
    week_input["nwp"]["ukv"]["zarr_path"] = str(ukv_store.resolve())
    week_input["nwp"]["ukv"].update(
        interval_start_minutes=-120, interval_end_minutes=120, time_resolution_minutes=60
    )
    week_input["solar_position"].update(
        interval_start_minutes=-120, interval_end_minutes=180, time_resolution_minutes=30
    )
    config_path = diagnostic_dir / "pvnet_2020_week_smoke_configuration.yaml"
    config_path.write_text(yaml.safe_dump(data_config, sort_keys=False))

    train_period = [(train_t0s[0].isoformat(sep=" "), train_t0s[-1].isoformat(sep=" "))]
    val_period = [(validation_t0s[0].isoformat(sep=" "), validation_t0s[-1].isoformat(sep=" "))]
    datamodule = PVNetDataModule(
        configuration=str(config_path),
        train_periods=train_period,
        val_periods=val_period,
        batch_size=8,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        seed=2727831,
    )
    datamodule.setup("fit")
    train_count = len(datamodule.train_dataset)
    val_count = len(datamodule.val_dataset)
    print(f"Satellite archive scans: {len(available_times)}; required scans: 2017; gaps preserved: 8", flush=True)
    print(f"Dropped incomplete satellite sample starts: {len(dropped_t0s)}", flush=True)
    print(f"Purged training starts for 180-minute forecast embargo: {len(purged_t0s)}", flush=True)
    print(f"PVNetDataset counts: {train_count} train; {val_count} validation", flush=True)
    print(f"Train t0 range: {train_t0s[0]} to {train_t0s[-1]}", flush=True)
    print(f"Validation t0 range: {validation_t0s[0]} to {validation_t0s[-1]}", flush=True)
    print(f"Last train target: {train_t0s[-1] + FORECAST_EMBARGO}; first validation t0: {validation_t0s[0]}", flush=True)
    if (train_count, val_count) != (len(train_t0s), len(validation_t0s)):
        raise ValueError(
            f"Sampler counts {(train_count, val_count)} differ from planned split "
            f"{(len(train_t0s), len(validation_t0s))}."
        )

    with MODEL_CONFIG_PATH.open() as stream:
        model_config = OmegaConf.create(yaml.safe_load(stream))
    model_config.model.location_id_mapping = {GSP_ID: 0}
    lightning_model = instantiate(model_config)
    batch = next(iter(datamodule.train_dataloader()))
    validate_batch_against_config(batch, lightning_model.model)
    print("Satellite batch:", tuple(batch["satellite_actual"].shape), flush=True)
    print("UKV batch:", tuple(batch["nwp"]["ukv"]["nwp"].shape), flush=True)
    print("Generation batch:", tuple(batch["generation"].shape), flush=True)
    print("Multimodal batch validation passed; starting the three-epoch CPU smoke test.", flush=True)

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    metric_history = MetricHistory()
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[metric_history],
        enable_model_summary=False,
        log_every_n_steps=10,
        default_root_dir=str(diagnostic_dir / "training_run"),
    )
    trainer.fit(
        lightning_model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )
    metrics = {
        "accelerator": "cpu",
        "epochs_completed": trainer.current_epoch,
        "optimizer_steps": trainer.global_step,
        "train_samples": train_count,
        "validation_samples": val_count,
        "satellite_missing_scans": 8,
        "purged_train_origins": len(purged_t0s),
        "epoch_metrics": metric_history.rows,
        "final_metrics": {
            name: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for name, value in trainer.callback_metrics.items()
            if np.isscalar(value) or (torch.is_tensor(value) and value.numel() == 1)
        },
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = diagnostic_dir / f"pvnet_2020_week_smoke_metrics_{run_id}.json"
    results_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"Training complete: {trainer.current_epoch} epochs, {trainer.global_step} optimizer steps", flush=True)
    print(f"Metrics JSON: {results_path}", flush=True)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
