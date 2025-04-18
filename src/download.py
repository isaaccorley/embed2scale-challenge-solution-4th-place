import argparse
import random
import time

from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset")
    parser.add_argument("--repo", type=str, default="embed2scale/SSL4EO-S12-downstream", help="Repository name")
    parser.add_argument("--directory", type=str, default="/data/data_eval", help="Local directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    done = False
    while not done:
        try:
            snapshot_download(
                repo_id=args.repo,
                repo_type="dataset",
                local_dir=args.directory,
                max_workers=args.workers,
                allow_patterns=["data_eval/s2l1c/*", "data_eval/s2l2a/*", "data_eval/s1/*"],
            )
            done = True
        except Exception as e:  # exponential backoff to avoid hammering the server
            print(e)
            wait = random.uniform(10, 30)  # random delay between 10-30s
            print(f"Retrying in {wait:.1f}s...")
            time.sleep(wait)
