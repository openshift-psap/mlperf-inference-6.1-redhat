#!/usr/bin/env python3
"""
MLflow Artifact Upload Script
=============================
Uploads a directory (recursively) as MLflow artifacts into a NEW run.

Key behaviors (fixed vs. your original):
- Uses MlflowClient for artifact logging (no fluent API run-context issues).
- Always sets mlflow.user (so UI doesn't show "Author: unknown").
- Always terminates the run (so lifecycle doesn't stay "created").
- Marks run FAILED if upload throws.

Usage:
  python3 upload_to_mlflow.py --experiment <experiment_name> --dir <directory>
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --tag submission=final,version=1.0
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --artifact-path submission
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --tracking-uri http://mlflow-server:5000
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("ERROR: mlflow is not installed. Please install it with: pip install mlflow")
    sys.exit(1)


def get_or_create_experiment(client: MlflowClient, experiment_name: str) -> str:
    """Get existing experiment or create a new one. Returns experiment_id."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating new experiment: {experiment_name}")
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    else:
        print(f"Using existing experiment: {experiment_name}")
    return experiment.experiment_id


def create_new_run(client: MlflowClient, experiment_id: str, run_name: Optional[str] = None) -> str:
    """Create a new MLflow run. Returns run_id."""
    print("Creating new run...")
    tags = {}
    if run_name:
        # MLflow recognizes mlflow.runName for display name
        tags["mlflow.runName"] = run_name

    run = client.create_run(experiment_id=experiment_id, tags=tags if tags else None)
    run_id = run.info.run_id
    print(f"Created new run: {run_id}")
    if run_name:
        print(f"  Run name: {run_name}")
    return run_id


def parse_tags(tag_string: str) -> Dict[str, str]:
    """Parse tags:
      - key1=value1,key2=value2
      - key1:value1,key2:value2
    """
    if not tag_string:
        return {}

    tags: Dict[str, str] = {}
    for pair in tag_string.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            key, value = pair.split("=", 1)
        elif ":" in pair:
            key, value = pair.split(":", 1)
        else:
            print(f"WARNING: Invalid tag format: {pair} (expected key=value or key:value)")
            continue
        tags[key.strip()] = value.strip()
    return tags


def resolve_user_tag() -> str:
    """Determine an author/user value for MLflow UI."""
    return (
        os.getenv("MLFLOW_TRACKING_USERNAME")
        or os.getenv("USER")
        or os.getenv("LOGNAME")
        or "unknown"
    )


def validate_directory(dir_path: str) -> Path:
    directory = Path(dir_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    return directory


def upload_directory(
    dir_path: str,
    experiment_name: str,
    artifact_path: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    """
    Creates a new run, sets tags, uploads artifacts, and terminates the run.

    Returns:
      run_id (str)
    """
    directory = validate_directory(dir_path)

    if artifact_path is None:
        artifact_path = directory.name

    # Configure tracking URI early (affects client + mlflow global config)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Using MLflow tracking URI: {tracking_uri}")
    else:
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_uri:
            mlflow.set_tracking_uri(env_uri)
            print(f"Using MLflow tracking URI (env): {env_uri}")
        else:
            print("Using default MLflow tracking URI (or local)")

    client = MlflowClient()

    print(f"Uploading directory {directory} to MLflow...")
    print(f"  Experiment: {experiment_name}")
    print(f"  Artifact Path: {artifact_path}")
    if run_name:
        print(f"  Run Name: {run_name}")

    # Merge tags + enforce mlflow.user
    merged_tags: Dict[str, str] = {}
    if tags:
        merged_tags.update(tags)
    merged_tags.setdefault("mlflow.user", resolve_user_tag())

    if merged_tags:
        print(f"  Tags: {merged_tags}")
    print()

    experiment_id = get_or_create_experiment(client, experiment_name)
    run_id = create_new_run(client, experiment_id, run_name=run_name)

    status = "FINISHED"
    try:
        # Set tags on the run (including mlflow.user)
        if merged_tags:
            print("Setting tags...")
            for k, v in merged_tags.items():
                client.set_tag(run_id, k, v)
                print(f"  Set tag: {k} = {v}")

        # Upload artifacts via client (robust for scripts/containers)
        print("Uploading directory contents...")
        client.log_artifacts(run_id, str(directory), artifact_path=artifact_path)

        print(f"✓ Successfully uploaded {directory.name} to run {run_id}")
        print(f"  Artifact path: {artifact_path}/")
        print(f"  Run ID: {run_id}")

    except Exception as e:
        status = "FAILED"
        print(f"ERROR: Failed to upload directory: {e}")
        raise
    finally:
        # Ensure run lifecycle is not left at "created"
        try:
            client.set_terminated(run_id, status=status)
            print(f"Run terminated with status: {status}")
        except Exception as term_e:
            print(f"WARNING: Failed to terminate run {run_id}: {term_e}")

    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a directory as MLflow artifacts into a NEW run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload directory to a new run in an experiment
  python3 upload_to_mlflow.py --experiment my_experiment --dir results

  # Upload with tags
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --tag submission=final,version=1.0

  # Upload with custom artifact path
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --artifact-path submission

  # Set tracking URI via flag (or env var MLFLOW_TRACKING_URI)
  python3 upload_to_mlflow.py --experiment my_experiment --dir results --tracking-uri http://mlflow-server:5000

  # Set author deterministically (optional)
  export MLFLOW_TRACKING_USERNAME=nmiriyal
  python3 upload_to_mlflow.py --experiment my_experiment --dir results
        """,
    )

    parser.add_argument("--experiment", required=True, help="MLflow experiment name")
    parser.add_argument("--dir", required=True, help="Path to directory to upload")
    parser.add_argument("--artifact-path", help="Artifact path in MLflow (default: directory name)")
    parser.add_argument(
        "--tag",
        "--mlflow-tag",
        dest="tags",
        help="MLflow tags in format key1=value1,key2=value2 or key1:value1,key2:value2",
    )
    parser.add_argument("--tracking-uri", help="MLflow tracking URI (or set MLFLOW_TRACKING_URI env var)")
    parser.add_argument("--run-name", help="Optional MLflow run display name")

    args = parser.parse_args()

    tags = parse_tags(args.tags) if args.tags else {}

    try:
        upload_directory(
            dir_path=args.dir,
            experiment_name=args.experiment,
            artifact_path=args.artifact_path,
            tags=tags,
            tracking_uri=args.tracking_uri,
            run_name=args.run_name,
        )
    except Exception:
        # keep traceback for debugging in automation runs
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
