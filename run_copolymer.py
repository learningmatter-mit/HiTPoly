"""
Build the box for one CSV row and run its MD in the same job (resumable).

    python run_copolymer.py --csv case_study_reruns_P5_P14.csv --row 3 --results ~/rerun
    python run_copolymer.py --csv systems.csv --row 0 --dry_run     # chain sizes / ion counts only

Rows whose box already exists (complete) reuse it; rows whose production run already
finished are skipped. A status CSV per row is written to <results>/status/<name>.csv.
Use run_copolymer_array.sh to submit every row as a SLURM array task.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.setrecursionlimit(5000)

from hitpoly.copolymer.pipeline import build_system, run_system, systems_from_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="systems CSV")
    p.add_argument("--row", type=int, required=True, help="0-based row index (SLURM_ARRAY_TASK_ID)")
    p.add_argument("--results", default=os.path.expanduser("~/results"), help="folder for box folders")
    p.add_argument("--platform", default="supercloud", choices=["supercloud", "local", "perlmutter"])
    p.add_argument("--build_only", action="store_true", help="build the box but do not run MD")
    p.add_argument("--dry_run", action="store_true", help="print chain sizes and ion counts, write nothing")
    p.add_argument("--polymer_count", type=int, default=None, help="override chains per box (testing)")
    p.add_argument("--simu_length", type=float, default=None, help="override production length in ns (testing)")
    args = p.parse_args()

    systems = systems_from_csv(args.csv)
    if not 0 <= args.row < len(systems):
        sys.exit(f"--row {args.row} out of range: {args.csv} has {len(systems)} rows")
    system, settings = systems[args.row]
    settings["platform"] = args.platform
    if args.polymer_count:
        settings["polymer_count"] = args.polymer_count
    if args.simu_length:
        settings["simu_length"] = args.simu_length
    paths = {"results_path": os.path.abspath(os.path.expanduser(args.results))}

    if args.dry_run:
        build_system(system, settings, paths, dry_run=True)
        return

    rec = run_system(
        system, settings, paths,
        simulate=not args.build_only,
        status_path=os.path.join(paths["results_path"], "status", f"{system['name']}.csv"),
    )
    print(rec)
    if rec["error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
