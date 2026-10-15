from huggingface_hub import snapshot_download
import os
from pathlib import Path

root = Path(__file__).parent.parent
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

models = [
    ("BAAI/bge-small-zh-v1.5", root / "models" / "bge-small-zh-v1.5"),
    ("BAAI/bge-reranker-v2-m3", root / "models" / "bge-reranker-v2-m3"),
]

for repo_id, local_dir in models:
    if local_dir.exists():
        print(f"[已存在] {local_dir.name}")
    else:
        print(f"[下载] {repo_id} → {local_dir.name}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"[完成] {local_dir.name}")
