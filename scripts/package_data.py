import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)


def main():
    parser = argparse.ArgumentParser(
        description="Package pre-built index data and upload to HuggingFace Hub"
    )
    parser.add_argument("--collection", default="dense_index",
                        help="Qdrant collection name (default: dense_index)")
    parser.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"),
                        help="HuggingFace dataset repo ID (e.g. your-org/rag-demo-data)")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY") or None)
    parser.add_argument("--db", default="data/chunks.db")
    parser.add_argument("--sparse-shards", default="data/sparse_shards")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    repo_id = args.repo
    if not repo_id:
        print("ERROR: HF_DATASET_REPO not set. Provide --repo or set the env var.")
        sys.exit(1)

    token = args.token
    if not token:
        print("ERROR: HF_TOKEN not set. Provide --token or set the env var.")
        sys.exit(1)

    db_path = Path(args.db)
    sparse_dir = Path(args.sparse_shards)

    if not db_path.exists():
        print(f"ERROR: chunks.db not found at {db_path}. Run index_data.py first.")
        sys.exit(1)

    shard_files = sorted(sparse_dir.glob("shard_*.pkl"))
    if not shard_files:
        print(f"ERROR: no sparse shards found in {sparse_dir}. Run index_data.py first.")
        sys.exit(1)

    from indexing import create_qdrant_snapshot, download_qdrant_snapshot

    qdrant_url = args.qdrant_url
    collection_name = args.collection

    print(f"Creating Qdrant snapshot for collection '{collection_name}' at {qdrant_url} ...")
    result = create_qdrant_snapshot(qdrant_url, collection_name, api_key=args.qdrant_api_key)
    snapshot_name = result["name"]
    print(f"  Snapshot created: {snapshot_name}")

    tmpdir = Path(tempfile.mkdtemp(prefix="qdrant_snapshot_"))
    snapshot_path = tmpdir / snapshot_name
    try:
        print(f"Downloading snapshot to {snapshot_path} ...")
        download_qdrant_snapshot(qdrant_url, collection_name, snapshot_name,
                                 str(snapshot_path), api_key=args.qdrant_api_key)
        print(f"  Downloaded ({snapshot_path.stat().st_size / 1024 / 1024:.1f} MB)")

        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)

        try:
            create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
        except Exception:
            pass

        print(f"Uploading chunks.db to {repo_id} ...")
        api.upload_file(
            path_or_fileobj=str(db_path),
            path_in_repo="chunks.db",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"Uploading {len(shard_files)} sparse shards to {repo_id}/sparse_shards/ ...")
        for i, sf in enumerate(shard_files):
            api.upload_file(
                path_or_fileobj=str(sf),
                path_in_repo=f"sparse_shards/{sf.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(shard_files)} shards uploaded")

        print(f"Uploading Qdrant snapshot to {repo_id} ...")
        api.upload_file(
            path_or_fileobj=str(snapshot_path),
            path_in_repo="dense_index.snapshot",
            repo_id=repo_id,
            repo_type="dataset",
        )

        print(f"\nPackage uploaded successfully to {repo_id}")
        print(f"  Files: chunks.db, sparse_shards/ ({len(shard_files)} shards), dense_index.snapshot")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
