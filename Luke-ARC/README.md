*(This was written entirely by Codex, on 23 Nov 2026, prompted by Luke)*

# PVNet on Oxford ARC HTC

This directory contains the HPC handoff helpers. The exploratory and plotting work remains in `Luke/`; data and run outputs on ARC live under `$DATA`, not in this Git repository.

## Planned first run

1. Push the code and notebook changes from the Mac, then clone/pull the repository on ARC. No local satellite or UKV data needs to be uploaded.
2. Connect to `htc-login.arc.ox.ac.uk` (through `gateway.arc.ox.ac.uk` when off the Oxford network) and use SLURM for compute. Login nodes are for light file operations and job submission only.
3. Start an interactive HTC allocation for environment setup and small checks. From the HTC login node:

   ```bash
   module load cluster/htc
   srun -p interactive --pty /bin/bash
   ```

   In that allocated shell, inspect available Python modules with `module avail Python` and choose Python 3.12 if available; PVNet supports Python 3.11–3.13 and the current smoke-test environment was Python 3.12.
4. Set persistent ARC paths. Replace `<project-data-path>` with the `$DATA` path ARC reports for your project:

   ```bash
   export PVNET_DATA_ROOT="<project-data-path>/pvnet"
   export PVNET_VENV="$PVNET_DATA_ROOT/venvs/pvnet-smoke"
   mkdir -p "$PVNET_DATA_ROOT"
   ```

5. From the repository root, install the locked PVNet environment plus the packages used by the satellite and UKV diagnostics:

   ```bash
   UV_PROJECT_ENVIRONMENT="$PVNET_VENV" uv sync --locked --no-default-groups --group arc-smoke
   ```

   `uv.lock` pins the repository dependency set. The `arc-smoke` group adds `cfgrib`, `eccodes`, `gcsfs`, `ocf-blosc2`, and `ipykernel`, at versions exercised by the current local smoke test. The GPU preflight below will tell us whether the locked PyTorch build works with ARC's driver and GPU; if it does not, adjust the environment based on the error and ARC's supported software rather than guessing a CUDA build now.

   If using ARC's Jupyter application, register this environment as a notebook kernel:

   ```bash
   "$PVNET_VENV/bin/python" -m ipykernel install --user --name pvnet-htc --display-name "PVNet (HTC)"
   ```

6. Run the small HTC GPU preflight from `Luke-ARC/`:

   ```bash
   cd Luke-ARC
   sbatch gpu_environment_check.sbatch
   ```

   Check the resulting `pvnet-envcheck-<job-id>.out`. It checks imports and whether PyTorch can see and use the allocated GPU. It does not download data or run training.

## Run the notebook without copying data from the Mac

After the environment preflight succeeds, return to a compute allocation before doing data work. The main notebook is `Luke/pvnet_2020_model_smoke_test.ipynb`. Its data root is controlled by `PVNET_DATA_ROOT`; if unset on the Mac, it retains the existing `Luke/data/to_gitignore` default.

Existing notebook outputs are preserved from the Mac run, so any `/Users/...` paths in those saved outputs are historical; the code itself now uses repo-relative configs and `PVNET_DATA_ROOT` for data.

The notebook's pre-staged, single-sample shortcut is optional. When those stores are absent, it skips that shortcut and continues to the week workflow. Before running the notebook top-to-bottom on a clean ARC data area, download the one UKV grid pilot file it needs to derive the 24×24 crop:

```bash
cd <repo-root>
export PVNET_DATA_ROOT="<project-data-path>/pvnet"
export PVNET_VENV="<project-data-path>/pvnet/venvs/pvnet-smoke"
source "$PVNET_VENV/bin/activate"
python Luke-ARC/download_ukv_pilot.py
```

The script prompts for a CEDA bearer token without echoing it unless `CEDA_TOKEN` is already set in the shell. It writes only to `$PVNET_DATA_ROOT/public_bucket_diagnostic/ukv_20200621_06/`. Do not put credentials in the notebook, a job script, or Git.

For training on an allocated HTC GPU, set `PVNET_ACCELERATOR=gpu` in the environment used to launch the notebook kernel. The notebook keeps `cpu` as its default so it continues to work on the Mac. The model-training cell is still a short three-epoch smoke test, not a production training job.

The first one-week data-preparation run may take longer than an interactive allocation. We should use notebook cells for exploration, coverage checks, and plots, then move long or restartable downloads and training into Python entry points submitted with `sbatch` before scaling to two years and ten GSPs. The notebook remains the readable workflow and diagnostic record; scripts will make long jobs restartable and scheduler-friendly.

## Prepare the June 2020 UKV week on HTC

`prepare_ukv_week.py` is the batch version of the notebook's UKV week-download/crop cell. It reads the checked-in sampler/model configs and the verified pilot GRIB, downloads the 56 required forecast cycles (Wholesale 1 and 2; 112 GRIBs total), crops all 17 channels to the GSP 199 24×24 grid, and writes the sampler Zarr to:

```text
$PVNET_DATA_ROOT/public_bucket_diagnostic/ukv_gsp199_20200621_week_sampler.zarr
```

It checkpoints each cycle as a compressed NetCDF crop, resumes from those crops and `.part` downloads after interruption, and deletes each raw GRIB only after its crop succeeds. The estimated source transfer is about 28 GB. The job uses HTC's `medium` partition for up to 24 hours and needs no GPU. A rerun after interruption reuses completed cycle crops and removes an incomplete temporary Zarr before rebuilding it from the crops. The script refuses to overwrite an existing final Zarr.

From the repository root on an HTC login node, load the same Python module used for the environment and set persistent paths. Pass the CEDA token through the environment without placing it in shell history or a script:

```bash
cd "$HOME/src/PVNet"
module load Python/3.12.3-GCCcore-13.3.0
export PVNET_DATA_ROOT="$DATA/pvnet"
export PVNET_VENV="$PVNET_DATA_ROOT/venvs/pvnet-smoke"
read -r -s -p "CEDA bearer token: " CEDA_TOKEN
printf '\n'
export CEDA_TOKEN
sbatch --export=ALL Luke-ARC/prepare_ukv_week.sbatch
unset CEDA_TOKEN
```

Monitor the job and read its log with:

```bash
squeue -u "$USER"
tail -f pvnet-ukv-week-<job-id>.out
sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode
```

When it completes, verify that the final Zarr exists under `$PVNET_DATA_ROOT`. Then proceed with the notebook's satellite preparation and alignment/training checks. Do not run the notebook's UKV download cell as well; it would target the same outputs.

## Storage and transfer

- Keep source code in the Git checkout; keep persistent datasets, crops, logs, and checkpoints under `$DATA`.
- `$HOME` is quota-limited. `$SCRATCH` and `$TMPDIR` are temporary and are removed when a job ends; copy outputs you need to `$DATA` before then.
- Prefer downloading source data directly on ARC and cropping it there. If an ARC source endpoint is unavailable, transfer only the small derived pilot store needed for diagnosis, not the raw multi-week source data.
- The existing notebook uses absolute paths only for repo-relative configuration files; data paths are rooted at `PVNET_DATA_ROOT`.

## Feedback loop

I cannot connect to ARC or inspect its filesystem. After each stage, send back the command you ran, the job output/error (for batch jobs, the `.out` file and `sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode`), plus `module list` and Python/PyTorch version details when diagnosing the environment. Never include CEDA tokens or passwords.
