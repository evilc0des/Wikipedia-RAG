import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap data from HuggingFace Hub for RAG pipeline"
    )
    parser.add_argument("--collection", default="dense_index",
                        help="Qdrant collection name (default: dense_index)")
    parser.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"),
                        help="HuggingFace dataset repo ID (e.g. your-org/rag-demo-data)")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY") or None)
    parser.add_argument("--db", default="data/chunks.db")
    parser.add_argument("--sparse-shards", default="data/sparse_shards")
    parser.add_argument("--snapshot", default="data/dense_index.snapshot")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-import even if data exists")
    parser.add_argument("--download-only", action="store_true",
                        help="Download data files but skip Qdrant snapshot import")
    args = parser.parse_args()

    repo_id = args.repo
    if not repo_id:
        print("ERROR: HF_DATASET_REPO not set. Provide --repo or set the env var.")
        sys.exit(1)

    token = args.token
    if not token:
        print("WARNING: HF_TOKEN not set. Attempting anonymous download (may fail for private repos).")

    db_path = Path(args.db)
    sparse_dir = Path(args.sparse_shards)
    snapshot_path = Path(args.snapshot)

    # --- Step 1: Download data files from HF ---

    needs_download = args.force or not (
        db_path.exists()
        and sparse_dir.exists()
        and list(sparse_dir.glob("shard_*.pkl"))
        and snapshot_path.exists()
    )

    if needs_download:
        from huggingface_hub import hf_hub_download, list_repo_files

        print(f"Fetching file list from {repo_id} ...")
        try:
            repo_files = list_repo_files(repo_id, repo_type="dataset", token=token)
        except Exception as e:
            print(f"ERROR: Failed to list repo files: {e}")
            sys.exit(1)

        print(f"  Found {len(repo_files)} files in repo")

        has_chunks = "chunks.db" in repo_files
        has_snapshot = "dense_index.snapshot" in repo_files
        sparse_files = [f for f in repo_files
                        if f.startswith("sparse_shards/") and f.endswith(".pkl")]

        print(f"  chunks.db: {'yes' if has_chunks else 'MISSING'}")
        print(f"  dense_index.snapshot: {'yes' if has_snapshot else 'MISSING'}")
        print(f"  sparse shards: {len(sparse_files)} files")

        if not has_chunks:
            print("ERROR: chunks.db not found in repo.")
            sys.exit(1)

        db_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading chunks.db ...")
        hf_hub_download(
            repo_id=repo_id, filename="chunks.db", repo_type="dataset",
            token=token, local_dir="data", local_dir_use_symlinks=False,
        )
        print(f"  Saved to {db_path}")

        if sparse_files:
            sparse_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {len(sparse_files)} sparse shards ...")
            for i, sf in enumerate(sparse_files):
                hf_hub_download(
                    repo_id=repo_id, filename=sf, repo_type="dataset",
                    token=token, local_dir="data", local_dir_use_symlinks=False,
                )
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(sparse_files)} shards downloaded")
            print(f"  {len(sparse_files)} sparse shards saved to {sparse_dir}")
        else:
            print("WARNING: No sparse shards found in repo.")

        if has_snapshot:
            print("Downloading dense_index.snapshot ...")
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo_id, filename="dense_index.snapshot", repo_type="dataset",
                token=token, local_dir="data", local_dir_use_symlinks=False,
            )
            print(f"  Saved to {snapshot_path}")
        else:
            print("ERROR: dense_index.snapshot not found in repo.")
            sys.exit(1)
    else:
        print(f"Data files already exist. Use --force to re-download.")

    if args.download_only:
        print("Download complete (--download-only). Skipping Qdrant snapshot import.")
        return

    # --- Step 2: Import snapshot into Docker Qdrant ---

    if not snapshot_path.exists():
        print(f"ERROR: Snapshot file not found at {snapshot_path}")
        sys.exit(1)

    qdrant_url = args.qdrant_url
    collection_name = args.collection

    from indexing import wait_for_qdrant, import_qdrant_snapshot

    print(f"Waiting for Qdrant at {qdrant_url} ...")
    try:
        wait_for_qdrant(qdrant_url)
        print("  Qdrant is healthy.")
    except TimeoutError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(url=qdrant_url, api_key=args.qdrant_api_key)

    if client.collection_exists(collection_name):
        point_count = client.count(collection_name=collection_name).count
        if point_count > 0 and not args.force:
            print(f"Collection '{collection_name}' already has {point_count} points. "
                  "Skipping snapshot import. Use --force to re-import.")
            client.close()
            return

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    client.close()

    print(f"Importing snapshot into Qdrant collection '{collection_name}' ...")
    try:
        import_qdrant_snapshot(
            qdrant_url, collection_name, str(snapshot_path),
            api_key=args.qdrant_api_key,
        )
        print("  Snapshot imported successfully.")
    except Exception as e:
        print(f"ERROR: Snapshot import failed: {e}")
        sys.exit(1)

    print("\nBootstrap complete. Data is ready.")


if __name__ == "__main__":
    main()
