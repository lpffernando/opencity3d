"""手动下载 SigLIP 模型到 HF 缓存目录（绕开 huggingface_hub 的 HEAD 元数据检查 bug）。

用法:
    python setup_hf_cache.py            # 只下载缺失文件
"""
import os
import sys
from pathlib import Path

import requests

# 需要下载的文件: (repo_id, filename)
FILES = [
    ("timm/ViT-SO400M-14-SigLIP-384", "open_clip_config.json"),
    ("timm/ViT-SO400M-14-SigLIP-384", "open_clip_pytorch_model.bin"),
    ("timm/ViT-SO400M-14-SigLIP-384", "open_clip_model.safetensors"),
    ("timm/ViT-SO400M-14-SigLIP-384", "tokenizer.json"),
    ("timm/ViT-SO400M-14-SigLIP-384", "tokenizer_config.json"),
    ("timm/ViT-SO400M-14-SigLIP-384", "special_tokens_map.json"),
    ("timm/ViT-B-16-SigLIP", "open_clip_config.json"),
    ("timm/ViT-B-16-SigLIP", "tokenizer.json"),
    ("timm/ViT-B-16-SigLIP", "tokenizer_config.json"),
    ("timm/ViT-B-16-SigLIP", "special_tokens_map.json"),
    ("timm/ViT-B-16-SigLIP", "open_clip_pytorch_model.bin"),
]

# 提交哈希（revision）
REVISIONS = {
    "timm/ViT-SO400M-14-SigLIP-384": "ac16108d567c4389e6cd2b11c9b8585f7474435b",
    "timm/ViT-B-16-SigLIP": "41f575766f40e752fdd1383e9565b7f02388c1c4",
}


def cache_root() -> Path:
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return root / "hub"


def repo_dir(repo_id: str) -> Path:
    return cache_root() / ("models--" + repo_id.replace("/", "--"))


def download(url: str, dst: Path, expected: int = 0) -> bool:
    if dst.exists() and dst.stat().st_size > 0 and (expected == 0 or dst.stat().st_size == expected):
        print(f"  [skip] {dst.name} ({dst.stat().st_size/1e6:.0f} MB)")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  [down] {dst.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dst)
    print(f"  [done] {dst.name} ({dst.stat().st_size/1e6:.0f} MB)")
    return True


def main():
    for repo_id, filename in FILES:
        rev = REVISIONS[repo_id]
        snapshot = repo_dir(repo_id) / "snapshots" / rev
        dst = snapshot / filename
        url = f"https://hf-mirror.com/{repo_id}/resolve/{rev}/{filename}"
        try:
            download(url, dst)
        except Exception as e:
            print(f"  [FAIL] {repo_id}/{filename}: {e}")
        # 写 refs/main，让 huggingface_hub 能找到 revision
        refs = repo_dir(repo_id) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(rev)
    print("\n缓存布局完成:")
    for repo_id in REVISIONS:
        print(f"  {repo_dir(repo_id)}")


if __name__ == "__main__":
    main()
