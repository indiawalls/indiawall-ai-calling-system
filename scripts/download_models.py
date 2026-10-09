"""
Download IndicConformer sherpa-onnx model files for Hindi STT.

Downloads INT8-quantized CTC model from HuggingFace:
    - hi/model.int8.onnx   (~150-200 MB)
    - tokens.txt           (~20 KB, shared across Indic languages)

Source: parismitaglobalsolutions/indicconformer-sherpa-onnx

Usage:
    python download_models.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-downloader")

# Model repository on HuggingFace
REPO_ID = "parismitaglobalsolutions/indicconformer-sherpa-onnx"

# Local directory to save model files
MODEL_DIR = os.environ.get("SHERPA_MODEL_DIR", "models/indicconformer-hi")

# Files to download for Hindi CTC model
MODEL_FILES = {
    "hi/model.int8.onnx": "model.int8.onnx",   # Hindi CTC model
    "tokens.txt": "tokens.txt",                  # Shared Indic vocabulary
}


def download_models(target_dir: str = None):
    """Download IndicConformer model files from HuggingFace."""
    dest_dir = target_dir or MODEL_DIR
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface-hub is not installed. Install it with:\n"
            "  pip install huggingface-hub"
        )
        sys.exit(1)

    os.makedirs(dest_dir, exist_ok=True)

    logger.info(f"Downloading IndicConformer Hindi CTC model from {REPO_ID}")
    logger.info(f"Saving to: {os.path.abspath(dest_dir)}")

    for repo_path, local_name in MODEL_FILES.items():
        local_path = os.path.join(dest_dir, local_name)

        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            logger.info(f"{local_name} already exists ({size_mb:.1f} MB), skipping")
            continue

        logger.info(f"  ↓ Downloading {repo_path} → {local_name}...")
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=repo_path,
                local_dir=dest_dir,
            )

            # hf_hub_download preserves repo directory structure
            # Move file to dest_dir root with the desired local_name
            if os.path.abspath(downloaded) != os.path.abspath(local_path):
                import shutil
                shutil.move(downloaded, local_path)
                # Clean up any empty subdirectories created by HF
                for dirpath, dirnames, filenames in os.walk(dest_dir, topdown=False):
                    if dirpath != dest_dir and not os.listdir(dirpath):
                        os.rmdir(dirpath)

            size_mb = os.path.getsize(local_path) / 1024 / 1024
            logger.info(f"  ✓ {local_name} downloaded ({size_mb:.1f} MB)")
        except Exception as e:
            logger.error(f"  ✗ Failed to download {repo_path}: {e}")
            sys.exit(1)

    # Verify all files exist
    logger.info("\nVerifying model files...")
    all_ok = True
    for repo_path, local_name in MODEL_FILES.items():
        local_path = os.path.join(dest_dir, local_name)
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            logger.info(f"  ✓ {local_name} ({size_mb:.1f} MB)")
        else:
            logger.error(f"  ✗ {local_name} MISSING")
            all_ok = False

    if all_ok:
        logger.info("\n✅ All model files downloaded successfully!")
        logger.info(f"   Model directory: {os.path.abspath(dest_dir)}")
        return True
    else:
        logger.error("\n❌ Some model files are missing. Please retry.")
        return False


def ensure_models_exist(target_dir: str = None) -> bool:
    """Check if all required model files exist; auto-download if missing."""
    dest_dir = target_dir or MODEL_DIR
    missing = []
    for repo_path, local_name in MODEL_FILES.items():
        local_path = os.path.join(dest_dir, local_name)
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            missing.append(local_name)

    if missing:
        logger.info(f"Local STT model files missing ({', '.join(missing)}). Auto-downloading...")
        download_models(dest_dir)
        return True
    else:
        logger.info(f"Local STT model verified at {os.path.abspath(dest_dir)}.")
        return False


if __name__ == "__main__":
    download_models()
