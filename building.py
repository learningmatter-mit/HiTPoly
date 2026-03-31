import argparse
import os
import shlex
import time
from pathlib import Path

import pandas as pd
import sys

sys.setrecursionlimit(5000)


def str_to_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def get_row_value(row, key, default=None):
    value = row[key] if key in row and pd.notna(row[key]) else default
    return value


def build_folder_name(row, simu_temp, charge_scale, charges, concentration, timestamp):
    poly_name = str(get_row_value(row, "polymer name", "polymer")).replace(" ", "_")
    repeats = int(get_row_value(row, "repeat units", 1))
    if concentration > 0:
        return (
            f"{poly_name}_N{repeats}_T{simu_temp}_C{concentration}_"
            f"q0{int(charge_scale * 100)}_q{charges}_{timestamp}"
        )
    return f"{poly_name}_N{repeats}_q{charges}_{timestamp}"


def resolve_salt_type(row, default_salt_type):
    if "salt_in_simu" in row and not str_to_bool(row["salt_in_simu"]):
        return None
    if "salt_type" in row and pd.notna(row["salt_type"]):
        salt_type = str(row["salt_type"]).strip()
        return None if salt_type == "None" else salt_type
    return default_salt_type


def create_slurm_script(script_path, log_dir, job_name, repo_path, command):
    script_contents = "\n".join(
        [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={log_dir / (job_name + '.out')}",
            f"#SBATCH --error={log_dir / (job_name + '.err')}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --time=24:00:00",
            "",
            f"cd {shlex.quote(str(repo_path))}",
            shlex.join(command),
            "",
        ]
    )
    script_path.write_text(script_contents)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Slurm scripts for builder_ligpargen_openmm.py jobs"
    )
    parser.add_argument(
        "--csv_path",
        default="data/polymer_folders_T415_LPG_250110.csv",
        help="CSV file describing the polymer jobs",
    )
    parser.add_argument(
        "--results_path",
        default=f"{os.path.expanduser('~')}/HiTPoly/results/LPG_COND_T415",
        help="Base directory where builder outputs will be written",
    )
    parser.add_argument(
        "--slurm_dir",
        required=True,
        help="Directory where generated Slurm scripts should be written",
    )
    parser.add_argument(
        "--simu_type",
        default="conductivity",
        help="Simulation type to request from the builder",
    )
    parser.add_argument(
        "--simu_length",
        default="100",
        help="Simulation length in ns",
    )
    parser.add_argument(
        "--md_save_time",
        default="12500",
        help="MD save interval",
    )
    parser.add_argument(
        "--platform",
        default="local",
        help="Builder platform argument",
    )
    parser.add_argument(
        "--system",
        default="polymer",
        help="Builder system argument",
    )
    parser.add_argument(
        "--salt_type",
        default="Li.TFSI",
        help="Default salt type when not provided by the CSV",
    )
    parser.add_argument(
        "--hitpoly_path",
        default=None,
        help="Optional explicit HiTPoly path to pass through",
    )
    parser.add_argument(
        "--hitpoly_env",
        default="hitpoly",
        help="Analysis environment name",
    )
    args = parser.parse_args()

    repo_path = Path(__file__).resolve().parent
    builder_path = repo_path / "builder_ligpargen_openmm.py"
    slurm_dir = Path(args.slurm_dir).expanduser().resolve()
    log_dir = slurm_dir / "logs"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results_path).expanduser().mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    timestamp = time.strftime("%y%m%d_H%HM%M")

    for ind, (_, row) in enumerate(df.iterrows()):
        smiles = str(get_row_value(row, "smiles", "")).strip()
        charge_scale = float(get_row_value(row, "charge scale", 0.75))
        charges = str(get_row_value(row, "charge", "LPG"))
        concentration = int(get_row_value(row, "salt amount", 0))
        simu_temp = int(get_row_value(row, "temperature", 430))
        solvent_count = int(get_row_value(row, "chains", 30))
        repeats = int(get_row_value(row, "repeat units", 1))
        add_end_carbons = str(get_row_value(row, "add_end_Cs", True))
        salt_type = resolve_salt_type(row, args.salt_type)

        folder_name = build_folder_name(
            row=row,
            simu_temp=simu_temp,
            charge_scale=charge_scale,
            charges=charges,
            concentration=concentration,
            timestamp=f"{timestamp}_{ind:04d}",
        )
        save_path = str(Path(args.results_path).expanduser() / folder_name)
        job_name = folder_name[:80]

        command = [
            "python",
            str(builder_path),
            "--save_path",
            save_path,
            "--results_path",
            save_path,
            "--smiles_path",
            smiles,
            "--solvent_count",
            str(solvent_count),
            "--repeats",
            str(repeats),
            "--concentration",
            str(concentration),
            "--charge_scale",
            str(charge_scale),
            "--charge_type",
            charges,
            "--temperature",
            str(simu_temp),
            "--simu_length",
            str(args.simu_length),
            "--md_save_time",
            str(args.md_save_time),
            "--platform",
            args.platform,
            "--system",
            args.system,
            "--simu_type",
            args.simu_type,
            "--add_end_Cs",
            add_end_carbons,
            "--hitpoly_env",
            args.hitpoly_env,
        ]
        if salt_type is None:
            command.extend(["--salt_type", "None"])
        else:
            command.extend(["--salt_type", salt_type])
        if args.hitpoly_path:
            command.extend(["--hitpoly_path", args.hitpoly_path])

        script_path = slurm_dir / f"{folder_name}.slurm"
        create_slurm_script(
            script_path=script_path,
            log_dir=log_dir,
            job_name=job_name,
            repo_path=repo_path,
            command=command,
        )
        print(f"WROTE {script_path}")


if __name__ == "__main__":
    main()
