from huggingface_hub import snapshot_download
import os
from pathlib import Path
import shutil

root = Path(__file__).parent.parent
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def check_model_complete(model_path, required_files):
    if not model_path.exists():
        return False
    for file in required_files:
        if not (model_path / file).exists():
            return False
    return True

qwen_path = root/"models"/"qwen2.5"
qwen_files = ["config.json", "model.safetensors", "tokenizer.json"]

if check_model_complete(qwen_path, qwen_files):
    print("qwen already\n")
else:
    if qwen_path.exists():
        shutil.rmtree(qwen_path)
    snapshot_download(
        repo_id="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        local_dir=str(qwen_path),
        resume_download=True,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.h5", "*.msgpack"],
    )

minilm_path = root / "models" / "all-MiniLM-L6-v2"
minilm_files = ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer.json"]

if check_model_complete(minilm_path, minilm_files):
    print("minilm already\n")
else:
    if minilm_path.exists():
        shutil.rmtree(minilm_path)
    snapshot_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        local_dir=str(minilm_path),
        resume_download=True,
        local_dir_use_symlinks=False,
    )

crossencoder_path = root / "models" / "ms-marco-MiniLM-L-6-v2"
cross_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]

if check_model_complete(crossencoder_path, cross_files):
    print("cross already\n")
else:
    if crossencoder_path.exists():
        shutil.rmtree(crossencoder_path)
    snapshot_download(
        repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        local_dir=str(crossencoder_path),
        resume_download=True,
        local_dir_use_symlinks=False,
    )


compress_model_path = root / "models" / "bart-base-chinese"
compress_files = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt"
]

if check_model_complete(compress_model_path, compress_files):
    print("compress already\n")
else:
    if compress_model_path.exists():
        shutil.rmtree(compress_model_path)
    snapshot_download(
        repo_id="google-bert/bert-base-chinese",
        local_dir=str(compress_model_path),
        resume_download=True,
        local_dir_use_symlinks=False,
    )